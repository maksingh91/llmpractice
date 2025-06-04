from transformers import pipeline, AutoTokenizer

def create_simple_llm(model_name="distilgpt2"):
    """
    Create a simple LLM using the Hugging Face Transformers library.
    This function initializes a text generation pipeline with the specified pre-trained model.
    Args:
        model_name (str): The name of the Hugging Face model to use.
    """
    # Load a pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a text generation pipeline
    llm = pipeline("text-generation", model=model_name, pad_token_id=50256)
    
    return llm

 
llm = create_simple_llm()
aprompt = "Once upon a time"
response = llm(aprompt, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])
    