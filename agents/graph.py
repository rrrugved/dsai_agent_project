import os
import re
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState
from agents.tools import (
    extract_text_from_image,
    transcribe_audio,
    fetch_webpage_content,
    fetch_youtube_transcript,
    parse_pdf
)
from agents.rag_builder import (
    ingest_into_qdrant,
    retrieve_from_qdrant,
    select_top_k_for_context,
    source_signatures,
)

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

def _detect_urls(text: str) -> List[str]:
    """Extracts all HTTP/HTTPS URLs from a given text string."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text)

def _is_youtube_url(url: str) -> bool:
    """Validates if a parsed URL belongs to the YouTube domain."""
    return "youtube.com" in url.lower() or "youtu.be" in url.lower()

def _get_last_user_message(messages: List[Any]) -> str:
    """Helper function to extract the text content of the last user message across turns."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _content_to_text(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    return ""


def _content_to_text(content: Any) -> str:
    """Normalize message content into plain text.

    Gemini/LangChain can sometimes return message content as a list of blocks
    instead of a plain string. This helper keeps the graph resilient to both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def ingest_and_classify(state: AgentState) -> Dict[str, Any]:
    """
    Classifies user intent using Gemini Flash. Resets per-turn state variables 
    to prevent state bleed in multi-turn conversations.
    """
    messages = state.get("messages", [])
    file_paths = state.get("file_paths", [])
    last_user_message = _get_last_user_message(messages)
    file_exts = {os.path.splitext(path)[1].lower() for path in file_paths}

    has_files = len(file_paths) > 0
    has_urls = len(_detect_urls(last_user_message)) > 0
    clean_prompt = last_user_message.strip()
    has_audio_only = any(ext in {".mp3", ".wav", ".m4a"} for ext in file_exts) and not any(
        ext in {".pdf", ".png", ".jpg", ".jpeg"} for ext in file_exts
    )

    base_state_reset = {
        "is_context_relevant": None,
        "extracted_text_map": {},
        "retrieved_context": None
    }

    if has_files or has_urls:
        intent_prompt = f"""
        Analyze the following user prompt which accompanies an uploaded file or link.
        
        Step 1: Determine if the prompt contains a specific, actionable instruction.
        If it is vague (e.g., "read this", "here", ""), the result is AMBIGUOUS.
        
        Step 2: If it is actionable, classify it into ONE of these strict categories based on its semantic meaning:
        - summarization (e.g., "summarize", "tl;dr", "give me the gist", "shorten this")
        - sentiment (e.g., "how does the author feel", "what is the tone", "sentiment")
        - code_explanation (e.g., "explain this code", "find the bug", "fix this script")
        - qa (For any general questions, data extraction, or anything else)

        Output ONLY in one of these exact formats:
        AMBIGUOUS
        VALID | <category>

        User Prompt: "{clean_prompt}"
        """
        
        intent_res = llm.invoke([HumanMessage(content=intent_prompt)])
        response_text = _content_to_text(intent_res.content).strip().upper()

        if "AMBIGUOUS" in response_text:
            return {
                **base_state_reset,
                "intent_type": "ambiguous",
                "needs_clarification": True,
                "plan_trace": ["Input received: Ambiguous instructions provided with media. Routing to Follow-up."],
                "task_category": None
            }
        
        task = "qa"
        if "VALID |" in response_text:
            try:
                task = response_text.split("|")[1].strip().lower()
            except IndexError:
                pass

        if has_audio_only and task in {"qa", "summarization"}:
            audio_keywords = ("summarize", "summary", "transcribe", "transcript", "what is said", "what does it say")
            if task == "summarization" or any(keyword in clean_prompt.lower() for keyword in audio_keywords):
                task = "audio_summary"
            
        return {
            **base_state_reset,
            "intent_type": "tool_required",
            "needs_clarification": False,
            "plan_trace": [f"Input received: Media/URL detected. Smart routing identified task: {task}"],
            "task_category": task
        }

    return {
        **base_state_reset,
        "intent_type": "conversational",
        "needs_clarification": False,
        "plan_trace": ["Input received: Standard text query. Routing to Direct LLM."],
        "task_category": "qa"
    }

def followup_node(state: AgentState) -> Dict[str, Any]:
    """Generates a clarifying question when the user prompt is ambiguous or extracted context is irrelevant."""
    file_paths = state.get("file_paths", [])
    is_relevant = state.get("is_context_relevant")
    intent = state.get("intent_type")
    
    if intent == "ambiguous" and file_paths:
        question = (
            f"I see you uploaded {len(file_paths)} file(s). Could you please clarify what you would like me to do with them? "
            "(e.g., summarize, extract action items, perform sentiment analysis, or explain code)"
        )
    elif is_relevant is False:
        question = (
            "I searched through the provided file(s), but I could not find "
            "relevant information to answer your request. Could you please clarify or ask a different question?"
        )
    else:
        question = "Could you please provide more details or clarify what task you would like me to perform?"

    trace = list(state.get("plan_trace", []))
    trace.append("Action: Prompted user for clarification.")
    return {
        "messages": [AIMessage(content=question)],
        "plan_trace": trace
    }

def direct_llm_node(state: AgentState) -> Dict[str, Any]:
    """Bypasses tool execution to directly answer conversational queries."""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    trace = list(state.get("plan_trace", []))
    trace.append("Action: Answered query directly using internal LLM knowledge.")
    return {
        "messages": [response],
        "plan_trace": trace
    }

def planner_and_tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes appropriate data extraction tools based on input file types and detected URLs."""
    file_paths = state.get("file_paths", [])
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)

    extracted_map = dict(state.get("extracted_text_map", {}))
    trace = list(state.get("plan_trace", []))

    for path in file_paths:
        filename = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            trace.append(f"Executing Tool: parse_pdf on '{filename}'")
            res = parse_pdf.invoke({"file_path": path})
            extracted_map[filename] = res

        elif ext in [".png", ".jpg", ".jpeg"]:
            trace.append(f"Executing Tool: extract_text_from_image (OCR) on '{filename}'")
            res = extract_text_from_image.invoke({"file_path": path})
            extracted_map[filename] = res

        elif ext in [".mp3", ".wav", ".m4a"]:
            trace.append(f"Executing Tool: transcribe_audio (Whisper) on '{filename}'")
            res = transcribe_audio.invoke({"file_path": path})
            extracted_map[filename] = res

    combined_corpus = last_user_message + " " + " ".join(extracted_map.values())


    found_urls = _detect_urls(combined_corpus)

    for url in found_urls:
        if _is_youtube_url(url):
            trace.append(f"Executing Tool: fetch_youtube_transcript for '{url}'")
            res = fetch_youtube_transcript.invoke(url)
            extracted_map[f"YouTube: {url}"] = res
            if res.startswith("Fallback:") or res.startswith("No YouTube URL detected"):
                trace.append(f"Warning: YouTube transcript could not be fetched: {res}")
            else:
                trace.append("Action: Successfully transcribed YouTube video and added to context map.")
        else:
            trace.append(f"Executing Tool: fetch_webpage_content for '{url}'")
            res = fetch_webpage_content.invoke(url)
            extracted_map[f"Web: {url}"] = res
            trace.append(f"Action: Successfully scraped webpage and added to context map.")

    return {
        "extracted_text_map": extracted_map,
        "plan_trace": trace
    }

def rag_ingestion_node(state: AgentState) -> Dict[str, Any]:
    """Takes extracted raw text, chunks it, and upserts vectors into Qdrant."""
    extracted_map = state.get("extracted_text_map", {})
    trace = list(state.get("plan_trace", []))

    if extracted_map:
        ingest_result = ingest_into_qdrant(extracted_map)
        added_chunks = ingest_result.get("added_chunks", 0)
        skipped_sources = ingest_result.get("skipped_sources", 0)
        if added_chunks:
            trace.append(f"Action: Chunked and ingested {added_chunks} vectors into Qdrant Database.")
        if skipped_sources:
            trace.append(f"Action: Skipped {skipped_sources} already-cached source(s).")
        if not added_chunks and not skipped_sources:
            trace.append("Action: No new chunks were created for Qdrant ingestion.")
    else:
        trace.append("Action: No text extracted; skipped Qdrant ingestion.")

    return {
        "extracted_text_map": extracted_map,
        "plan_trace": trace
    }

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Embeds user query and fetches the top relevant text chunks from Qdrant."""
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    trace = list(state.get("plan_trace", []))
    extracted_map = dict(state.get("extracted_text_map", {}))

    top_k = select_top_k_for_context(extracted_map)
    current_source_signatures = source_signatures(extracted_map)
    retrieved_text = retrieve_from_qdrant(
        last_user_message,
        top_k=top_k,
        allowed_source_signatures=current_source_signatures,
    )
    trace.append(f"Action: Retrieval depth selected dynamically (top_k={top_k}).")
    
    if retrieved_text:
        trace.append("Action: Successfully retrieved relevant context from Qdrant.")
    else:
        trace.append("Action: No relevant context found in Qdrant.")

    return {
        "extracted_text_map": extracted_map,
        "retrieved_context": retrieved_text,
        "plan_trace": trace
    }

def relevancy_checker_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates whether retrieved vector chunks contain sufficient information to answer the query."""
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    retrieved_context = state.get("retrieved_context", "")
    
    trace = list(state.get("plan_trace", []))

    if not retrieved_context:
        trace.append("Self-Reflection Check: No extracted context found.")
        return {
            "is_context_relevant": False,
            "extracted_text_map": state.get("extracted_text_map", {}),
            "retrieved_context": retrieved_context,
            "plan_trace": trace
        }

    # A request to summarize/transcribe an uploaded source is intrinsically
    # grounded in that source.  Asking an LLM whether a short greeting answers
    # "summarize this audio" is both unnecessary and unstable; it can return
    # NO even though the transcript is exactly what must be summarized.
    task = state.get("task_category")
    if task == "audio_summary":
        trace.append("Self-Reflection Check: Current audio transcript is valid context for the requested summary.")
        return {
            "is_context_relevant": True,
            "extracted_text_map": state.get("extracted_text_map", {}),
            "retrieved_context": retrieved_context,
            "plan_trace": trace,
        }

    checker_prompt = f"""
    You are a strict relevance grader. Evaluate whether the extracted context below contains sufficient or relevant information to answer or address the user query.

    USER QUERY: {last_user_message}

    EXTRACTED CONTEXT (From Vector Search):
    {retrieved_context}

    Respond with ONLY 'YES' if the context is relevant and useful, or 'NO' if it is irrelevant, corrupt, or empty.
    """

    res = llm.invoke([HumanMessage(content=checker_prompt)])
    clean_answer = _content_to_text(res.content).strip().upper()
    is_relevant = clean_answer.startswith("YES")

    trace.append(f"Self-Reflection Check: Context relevance evaluated as {is_relevant}.")
    return {
        "is_context_relevant": is_relevant,
        "extracted_text_map": state.get("extracted_text_map", {}),
        "retrieved_context": retrieved_context,
        "plan_trace": trace
    }

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the final response formatted strictly according to task-specific guidelines."""
    task = state.get("task_category", "qa")
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    retrieved_context = state.get("retrieved_context", "")

    context_str = retrieved_context

    formatting_instructions = _build_formatting_instructions(task)

    system_instructions = f"""
    You are an expert multi-modal AI agent. Answer the user request using the provided context.
    All outputs must be text-only.

    {formatting_instructions}

    Context:
    {context_str}
    """

    full_messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=last_user_message)
    ]

    response = llm.invoke(full_messages)
    final_text = _content_to_text(response.content)

    trace = list(state.get("plan_trace", []))
    trace.append("Action: Synthesized final structured answer.")
    return {
        "extracted_text_map": state.get("extracted_text_map", {}),
        "retrieved_context": retrieved_context,
        "messages": [AIMessage(content=final_text)],
        "plan_trace": trace
    }


def _build_formatting_instructions(task: str) -> str:
    if task == "summarization":
        return """
        You MUST structure your output strictly as follows:
        1. **1-line Summary:** <One concise sentence summarizing the core content>
        2. **Key Highlights:**
           - <Bullet 1>
           - <Bullet 2>
           - <Bullet 3>
        3. **Detailed Summary:** <Exactly 5 sentences in one paragraph>
        """
    if task == "audio_summary":
        return """
        You MUST structure your output strictly as follows:
        1. **Transcription:** <Clean transcript of the audio>
        2. **1-line Summary:** <One concise sentence summarizing the audio>
        3. **Key Highlights:**
           - <Bullet 1>
           - <Bullet 2>
           - <Bullet 3>
        4. **Detailed Summary:** <Exactly 5 sentences in one paragraph>
        """
    if task == "sentiment":
        return """
        You MUST structure your output strictly as follows:
        - **Sentiment Label:** <Positive / Negative / Neutral / Mixed>
        - **Confidence Score:** <0% to 100%>
        - **Justification:** <One clear sentence explaining the reasoning>
        """
    if task == "code_explanation":
        return """
        You MUST structure your output strictly as follows:
        - **Code Functionality:** <Detailed breakdown of what the code does>
        - **Bug Analysis & Warnings:** <Identified bugs, syntax issues, or logical flaws (or state 'None detected')>
        - **Time & Space Complexity:** <Estimated Big-O complexity analysis>
        """
    return ""


def route_after_classify(state: AgentState) -> Literal["followup_node", "direct_llm_node", "planner_and_tool_node"]:
    """Determines the next graph node based on the classified intent type."""
    intent = state.get("intent_type")
    if intent == "ambiguous":
        return "followup_node"
    elif intent == "conversational":
        return "direct_llm_node"
    return "planner_and_tool_node"

def route_after_relevancy(state: AgentState) -> Literal["synthesizer_node", "followup_node"]:
    """Routes execution to synthesis or follow-up based on context relevancy."""
    if not state.get("retrieved_context") or state.get("is_context_relevant") is not True:
        return "followup_node"
    return "synthesizer_node"



builder = StateGraph(AgentState)

builder.add_node("ingest_and_classify", ingest_and_classify)
builder.add_node("followup_node", followup_node)
builder.add_node("direct_llm_node", direct_llm_node)
builder.add_node("planner_and_tool_node", planner_and_tool_node)
builder.add_node("rag_ingestion_node", rag_ingestion_node)
builder.add_node("retrieval_node", retrieval_node)
builder.add_node("relevancy_checker_node", relevancy_checker_node)
builder.add_node("synthesizer_node", synthesizer_node)

builder.add_edge(START, "ingest_and_classify")

builder.add_conditional_edges(
    "ingest_and_classify",
    route_after_classify,
    {
        "followup_node": "followup_node",
        "direct_llm_node": "direct_llm_node",
        "planner_and_tool_node": "planner_and_tool_node"
    }
)

builder.add_edge("planner_and_tool_node", "rag_ingestion_node")
builder.add_edge("rag_ingestion_node", "retrieval_node")
builder.add_edge("retrieval_node", "relevancy_checker_node")

builder.add_conditional_edges(
    "relevancy_checker_node",
    route_after_relevancy,
    {
        "synthesizer_node": "synthesizer_node",
        "followup_node": "followup_node"
    }
)

builder.add_edge("followup_node", END)
builder.add_edge("direct_llm_node", END)
builder.add_edge("synthesizer_node", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
