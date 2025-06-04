from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_simple_llm(model_name="distilgpt2"):
    """
    Create a simple LLM using the Hugging Face Transformers library.
    This function initializes a text generation pipeline with the specified pre-trained model.
    
    Args:
        model_name (str): The name of the Hugging Face model to use.
    """
    # Load a pre-trained model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a text generation pipeline
    llm = pipeline("text-generation", model=model_name, pad_token_id=50256)
    
    return llm

def explain():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Once upon a time"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded text: {decoded}")
    
explain()