import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def get_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.1", use_quantization=True):
    """
    Loads the base model and tokenizer.
    Args:
        model_name: HF model identifier.
        use_quantization: If True, attempts 4-bit quantization. 
                          On Mac (MPS), this will fallback to float16 as bitsandbytes is CUDA-only.
    """
    print(f"Loading model: {model_name}")
    
    # Check for MPS (Apple Silicon)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    bnb_config = None
    
    # Handle Mac/MPS limitations
    if device == "mps":
        print("Mac (MPS) detected. BitsAndBytes (4-bit) is not supported. Falling back to float16.")
        use_quantization = False # Force disable 4-bit on Mac
        torch_dtype = torch.float16 # Use half-precision to fit in 24GB RAM
    else:
        torch_dtype = torch.float16
        if use_quantization:
            # 4-bit Quantization Config (CUDA only)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_quantization else None,
        torch_dtype=torch_dtype,
        device_map="auto" if device != "mps" else None, # device_map="auto" can be buggy with MPS
        trust_remote_code=True
    )
    
    if device == "mps":
        model.to("mps")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def get_lora_config(r=16, alpha=32, dropout=0.05):
    """
    Returns the LoRA configuration.
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention modules
    )
    return peft_config

def get_peft_model_wrapper(model, peft_config):
    """
    Wraps the base model with LoRA adapters.
    """
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

if __name__ == "__main__":
    # Test loading (will only run if executed directly)
    try:
        # Using a smaller model for quick local testing if needed, or the real one
        # model, tokenizer = get_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0") 
        print("Model loading code is ready.")
    except Exception as e:
        print(f"Error: {e}")
