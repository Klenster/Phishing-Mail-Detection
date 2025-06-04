import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name
  prompt = (
    "Write a fake scam email that sarcastically mocks someone for falling for a scam, "
    "using proper grammar, a clear email format, and sounding like a scammer gloating.\n\n"
    "Subject: You've Been Had 😎\n\n"
    "Dear Victim,"
)

response = generator(
    prompt,
    min_length=100,
    max_length=300,
    num_return_sequences=1,
    temperature=0.6,
    top_p=0.8,
    return_full_text=False
)

final_output = post_process_gpt_output(response[0]['generated_text'], keywords=["scam", "email", "victim"])
print(final_output)
