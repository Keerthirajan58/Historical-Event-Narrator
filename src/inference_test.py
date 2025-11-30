import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini for Judge
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def load_model_for_inference(base_model_name, adapter_path):
    print(f"Loading base model: {base_model_name}")
    
    # Check for MPS
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Base Model (float16 for Mac)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device != "mps" else None,
        trust_remote_code=True
    )
    
    if device == "mps":
        model.to("mps")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA Adapter
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer, device

def generate_story(model, tokenizer, device, topic):
    prompt = f"Narrate this historical event with a creative twist.\n\nTopic: {topic}\n\nNarrative:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=800, 
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_with_gemini(topic, story):
    """
    Uses Gemini to judge the quality of the generated story.
    """
    if "GEMINI_API_KEY" not in os.environ:
        return "Gemini API Key not found. Skipping Judge evaluation."
        
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert literary critic and historian. Evaluate the following historical narrative based on these criteria:
    1. Creativity (1-10): How engaging and novel is the storytelling?
    2. Historical Grounding (1-10): Does it start with actual historical context?
    3. The Twist (1-10): Is there a compelling "what-if" or narrative twist?
    
    Topic: {topic}
    Generated Story:
    {story}
    
    Output format:
    Creativity: [Score]/10
    History: [Score]/10
    Twist: [Score]/10
    Explanation: [Brief explanation]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini Judge: {e}"

if __name__ == "__main__":
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    ADAPTER_PATH = "./results"
    
    model, tokenizer, device = load_model_for_inference(BASE_MODEL, ADAPTER_PATH)
    
    print("\nModel Loaded! Enter a historical topic to generate a story (or 'q' to quit).")
    
    while True:
        topic = input("\nEnter Topic: ")
        if topic.lower() == 'q':
            break
            
        story = generate_story(model, tokenizer, device, topic)
        
        print("\n" + "="*50)
        print(f"GENERATED STORY FOR: {topic}")
        print("="*50)
        print(story)
        print("="*50 + "\n")
        
        print("--- Gemini Judge Evaluation ---")
        judge_feedback = evaluate_with_gemini(topic, story)
        print(judge_feedback)
        print("-" * 30)
