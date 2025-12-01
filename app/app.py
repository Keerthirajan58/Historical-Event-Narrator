import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Historical Event Narrator",
    page_icon="📜",
    layout="wide"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 20px;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff914d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    .story-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        font-family: 'Georgia', serif;
        line-height: 1.6;
        font-size: 18px;
    }
    .judge-box {
        background-color: #1c1e24;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">📜 Historical Event Narrator</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #aaa;'>Reimagining History with AI</h3>", unsafe_allow_html=True)

# Sidebar for Controls
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Length", 200, 1000, 600)
    temperature = st.slider("Creativity (Temp)", 0.1, 1.5, 0.8)
    
    st.divider()
    st.info("Model: Mistral-7B + LoRA Adapter")
    st.info("Device: Mac (MPS)")

# Load Model Function (Cached)
@st.cache_resource
def load_model():
    base_model_name = "mistralai/Mistral-7B-v0.1"
    adapter_path = "./results"
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device != "mps" else None,
        trust_remote_code=True
    )
    
    if device == "mps":
        model.to("mps")
        
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer, device

# Judge Function
def evaluate_story(topic, story):
    if "GEMINI_API_KEY" not in os.environ:
        return None
    
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Evaluate this historical story.
    Topic: {topic}
    Story: {story}
    
    Provide scores (1-10) for:
    1. Creativity
    2. Historical Accuracy
    3. Twist Quality
    
    Keep it brief.
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "Judge unavailable."

# Custom Stopping Criteria
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Main Interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Enter a Topic")
    topic = st.text_input("Event Name", placeholder="e.g. The Fall of Rome")
    generate_btn = st.button("Generate Story ✨")

if generate_btn and topic:
    with col2:
        with st.spinner("Loading Time Machine... (Model Loading)"):
            try:
                model, tokenizer, device = load_model()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
        
        with st.spinner("Writing History..."):
            # Phase 1: Generate the Main Narrative
            # We ask ONLY for the narrative first
            prompt = f"Narrate the historical event '{topic}'.\n\nNarrative:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Stop if it tries to start the twist or a new topic
            stop_words = ["Topic:", "[What If Twist]"]
            stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
            # Add newline as a soft stop if needed, but let's stick to explicit headers
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_ids)])
                )
            
            story_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt
            story = story_raw.replace(prompt, "").strip()
            
            # Clean up if it generated the stop tokens
            for w in stop_words:
                story = story.split(w)[0].strip()

            # Phase 2: Force the Twist
            # We manually append the header and a strong starter
            twist_header = "\n\n[What If Twist]\n"
            twist_starter = "However, imagine a different outcome. What if"
            full_context = prompt + story + twist_header + twist_starter
            
            inputs_twist = tokenizer(full_context, return_tensors="pt").to(device)
            
            with st.spinner("Injecting Temporal Anomaly... (Generating Twist)"):
                with torch.no_grad():
                    outputs_twist = model.generate(
                        **inputs_twist, 
                        max_new_tokens=200, 
                        temperature=0.95, # Higher creativity for twist
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
            
            final_output = tokenizer.decode(outputs_twist[0], skip_special_tokens=True)
            
            # Extract just the story + twist (remove prompt)
            final_display = final_output.replace(prompt, "").strip()
            
            st.markdown(f'<div class="story-box">{final_display}</div>', unsafe_allow_html=True)
            
            # Judge
            with st.spinner("The Judge is deliberating..."):
                feedback = evaluate_story(topic, final_display)
                if feedback:
                    st.markdown(f'<div class="judge-box"><b>⚖️ AI Judge Verdict:</b><br>{feedback}</div>', unsafe_allow_html=True)
            
            # Cleanup Memory
            if device == "mps":
                torch.mps.empty_cache()
