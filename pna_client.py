import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline
import spaces

class PNAAssistantClient:
    def __init__(self, model_id="NurseCitizenDeveloper/nursing-llama-3-8b-fons"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Diversity Emojis from instructions
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]

    def _load_model(self):
        if self.model is None:
            print(f"Loading model {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

    @spaces.GPU()
    def generate_response(self, prompt, context="", history=[]):
        self._load_model()
        
        system_prompt = f"""You are a Professional Nurse Advocate (PNA) AI tutor. 
Your goal is to guide users in understanding the PNA role and the A-EQUIP model (Normative, Formative, Restorative, Personal Action).
You focus heavily on Restorative Supervision.

CONSTRAINTS:
1. Diversity: Always include one of these emojis in every response: {', '.join(self.diversity_emojis)}.
2. Pedagogical Style: Use open-ended questions. Avoid giving immediate answers. Guide the user to reflect.
3. Content Scope: Only assist with PNA, A-EQUIP, or listed nursing fields.
4. Voice: Maintain the person-centred, compassionate tone you were trained on.
5. Formatting: Max 2 short paragraphs or 6 bullet points.

CONTEXT FROM A-EQUIP GUIDE:
{context}
"""

        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return response.strip()
