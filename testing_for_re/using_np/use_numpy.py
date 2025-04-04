import numpy as np
import torch

print("Hello, numpy & torch!")
print("NumPy version:", np.__version__)
print("Torch version:", torch.__version__)

# Create two 768-dim embedding vectors (typical for language models)
embedding1 = np.random.normal(0, 0.1, 768)
embedding2 = np.random.normal(0, 0.1, 768)

# Normalize vectors (standard practice for embeddings)
embedding1 = embedding1 / np.linalg.norm(embedding1)
embedding2 = embedding2 / np.linalg.norm(embedding2)

# Calculate cosine similarity (dot product of normalized vectors)
similarity = np.dot(embedding1, embedding2)

print("Embedding shapes:", embedding1.shape, embedding2.shape)
print("Cosine similarity:", similarity)
