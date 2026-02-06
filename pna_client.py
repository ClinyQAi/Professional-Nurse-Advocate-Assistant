# CRITICAL: spaces MUST be imported FIRST before torch/CUDA
import spaces
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PNAAssistantClient:
    # Using user's merged MedGemma model - trained on person-centred language
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        # Don't set device in __init__ as it might be CPU on startup
        self.tokenizer = None
        self.model = None
        
        # Diversity Emojis from PNA instructions
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]

    def _load_model(self):
        if self.model is None:
            print(f"Loading model {self.model_id}...")
            # Inside @spaces.GPU, torch.cuda.is_available() should be True
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self.device = device
            print("Model loaded successfully!")

    @spaces.GPU()
    def generate_response(self, prompt, context="", history=[]):
        self._load_model()
        
        system_prompt = f"""You are a Professional Nurse Advocate (PNA) AI tutor. Your role is to guide nursing professionals through the A-EQUIP model (Advocating and Educating for Quality Improvement).

**Your Core Functions (A-EQUIP):**
1. Normative: Monitoring, evaluation, quality control
2. Formative: Education and development
3. Restorative: Clinical supervision (your primary focus)
4. Personal Action: Quality improvement

**Communication Style:**
- Use person-centred, compassionate language
- Always include a diversity emoji: {', '.join(self.diversity_emojis)}
- Ask open-ended questions before giving answers
- Focus on reflection and restorative supervision
- Keep responses to 2 short paragraphs or 6 bullet points max

**Scope:**
- Only discuss PNA, A-EQUIP, nursing fields
- For out-of-scope topics: "I can only assist with topics related to the Professional Nurse Advocate role and the A-EQUIP model."

**Reference Material:**
{context}
"""

        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nUser question: {prompt}"}
        ]
        
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.device)
        attention_mask = torch.ones_like(inputs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                attention_mask=attention_mask,
                max_new_tokens=300, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip()
