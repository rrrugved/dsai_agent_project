import uuid
import sys
from langchain_core.messages import HumanMessage
from agents.graph import graph

MAX_EXTRACTED_CHARS = 4000

def _format_extracted_text(result: dict) -> str:
    """Render extracted sources in a readable, demo-friendly format."""
    extracted_map = result.get("extracted_text_map", {}) or {}
    if not extracted_map:
        return "  > No extracted text available."

    sections = []
    for source, content in extracted_map.items():
        if not content:
            continue
        preview = content[:MAX_EXTRACTED_CHARS]
        if len(content) > MAX_EXTRACTED_CHARS:
            preview += "\n...[truncated for display]"
        sections.append(f"--- {source} ---\n{preview}")

    return "\n\n".join(sections) if sections else "  > No extracted text available."

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("Welcome to the Multi-Modal Agent Terminal!")
    print("Type 'quit' or 'exit' to stop the conversation.")
    print("=" * 60)
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:

        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye! ")
            break
            
        if not user_input:
            continue
            
        file_input = input("📎 Attach files (comma-separated paths, or press Enter to skip): ").strip()
        file_paths = [f.strip() for f in file_input.split(',')] if file_input else []

        state_input = {
            "messages": [HumanMessage(content=user_input)],
            "file_paths": file_paths
        }
        
        print("\n⏳ Agent is thinking...")
        print("-" * 60)
        
        try:
            result = graph.invoke(state_input, config)
            
            print("🧠 Internal Plan Trace:")
            for step in result.get("plan_trace", []):
                print(f"  > {step}")
            print("-" * 60)

            print("Extracted Text:")
            print(_format_extracted_text(result))
            print("-" * 60)
            
            final_message = result["messages"][-1].content
            print(f"Agent:\n{final_message}\n")
            
        except Exception as e:
            print(f"\n An error occurred: {e}")

if __name__ == "__main__":
    main()
