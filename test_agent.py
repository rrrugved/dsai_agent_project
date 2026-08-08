import uuid
from langchain_core.messages import HumanMessage
from agents.graph import graph

def run_test():
    print("Agent initialized! Type 'quit' to exit.")
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    file_paths = [] 

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "file_paths": file_paths
        }
        
        print("Agent is thinking...\n")
        
        result = graph.invoke(inputs, config=config)
        
        final_message = result["messages"][-1].content
        print(f" Agent: {final_message}")
        
        print("\n--- 🔍 Internal Plan Trace ---")
        for step in result.get("plan_trace", []):
            print(f" -> {step}")
        print("------------------------------")

if __name__ == "__main__":
    run_test()