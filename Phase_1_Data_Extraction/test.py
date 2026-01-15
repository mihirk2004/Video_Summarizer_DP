import numpy as np
import torch
import nltk
from transformers import pipeline

print("Testing imports...")

# Test PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# Test NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
text = "This is a test sentence for NLTK tokenization."
print(f"NLTK sentence tokenization: {sent_tokenize(text)}")

# Test transformers
print("Testing transformers...")
classifier = pipeline("sentiment-analysis", device=-1)  # Use CPU
result = classifier("I love this project!")[0]
print(f"Transformers sentiment: {result}")

print("✅ All imports successful!")