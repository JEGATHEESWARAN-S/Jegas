from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaChatbot:
    def __init__(self, model_name="meta-llama/Llama-2-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_response(self, user_message, max_length=100):
        inputs = self.tokenizer(user_message, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
