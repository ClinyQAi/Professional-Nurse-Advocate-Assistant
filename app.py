import gradio as gr
from knowledge_base import PNAKnowledgeBase
from pna_client import PNAAssistantClient
import os

# --- Constants ---
GUIDE_FILENAME = "Professional nurse advocate A-EQUIP model Guide.md"

# --- Initialize Components ---
kb = PNAKnowledgeBase(GUIDE_FILENAME)
client = PNAAssistantClient()

def chat_response(message, history):
    # 1. Search Knowledge Base
    context = kb.search(message)
    
    # 2. Generate Response using fine-tuned model
    # Note: history in Gradio is list of [user, bot] pairs
    response = client.generate_response(message, context=context, history=history)
    
    return response

# --- UI Layout ---
CSS = """
.gradio-container { max-width: 800px !important; margin: auto !important; }
.header { text-align: center; padding: 20px; border-bottom: 1px solid #eee; }
.title { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
"""

with gr.Blocks(css=CSS, title="PNA Assistant") as demo:
    with gr.Row(elem_classes="header"):
        gr.HTML("""
            <div class="title">👨🏾‍⚕️ Professional Nurse Advocate Assistant 👩🏽‍⚕️</div>
            <p>Guiding you through the A-EQUIP model and Restorative Supervision</p>
        """)
    
    chat = gr.ChatInterface(
        fn=chat_response,
        examples=["What is the A-EQUIP model?", "Tell me about Restorative Supervision", "What does a PNA do?"],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()
