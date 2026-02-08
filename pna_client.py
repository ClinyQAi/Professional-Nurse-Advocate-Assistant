"""
PNA Assistant Client - Using HF Inference API
No GPU required - runs on CPU Basic
"""
import os
import time
from huggingface_hub import InferenceClient

# List of models to try in order
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

class PNAAssistantClient:
    def __init__(self):
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        # Initialize with the first model, but we'll switch if needed
        self.current_model = MODELS[0]
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        print(f"PNA Assistant initialized. Candidate models: {MODELS}")

    def generate_response(self, prompt, context="", history=[]):
        emoji_list = ", ".join(self.diversity_emojis)
        system_content = f"You are a Professional Nurse Advocate (PNA) AI tutor. Include emoji: {emoji_list}. Keep responses to 2 paragraphs max. Context: {context}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Try each model in sequence
        for model in MODELS:
            try:
                print(f"Trying model: {model}...")
                response = self.client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error with {model}: {e}")
                continue
                
        return "I apologize, but I am experiencing technical difficulties with all available models. Please try again later."
