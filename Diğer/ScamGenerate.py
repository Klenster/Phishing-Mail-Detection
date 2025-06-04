import torch
from transformers import pipeline

# Load GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B',device=0,torch_dtype=torch.float16)

# Generate text
prompt = "Tell someone how stupid they are for falling for a scam email. Make it sound like a scam email."
response = generator(prompt,min_length=100,max_length=500,  # Increase length for full email
                     num_return_sequences=1,  # Generate multiple variations
                     temperature=0.8,  # Increase randomness
                     top_p=0.9)  # Use nucleus sampling for diversity)

print(response[0]['generated_text'])
