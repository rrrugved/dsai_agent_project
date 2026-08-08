from agents.graph import graph

image_data = graph.get_graph().draw_mermaid_png()

with open("graph_architecture2.png", "wb") as f:
    f.write(image_data)

print("✅ Updated Graph successfully saved as graph_architecture2.png!")