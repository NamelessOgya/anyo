# authenticate_hf.py
from huggingface_hub import login

print("Please enter your Hugging Face Hub access token:")
token = input()
login(token=token)
print("Successfully authenticated with Hugging Face Hub.")
