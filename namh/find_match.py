import numpy as np
import pandas as pd
from vectorize_pd import encode
from namh import find_closest_match

# Load data
data = pd.read_csv('data_vector.csv')

# Convert embedding vectors to numpy arrays
embedding_list = [np.array(eval(embedding)).flatten() for embedding in data['embedding']]

# Query similarity
input_text = "Clean makeup."
prompt_vector = encode(input_text)
prompt_vector = np.array(prompt_vector).flatten()

# Check embedding_list sizing
for vec in embedding_list:
    assert len(vec) == len(prompt_vector), f"Vector dimensions mismatch: {len(vec)} != {len(prompt_vector)}"

# Run the actual code
closest_match, highest_similarity, community_vectors = find_closest_match(prompt_vector, embedding_list)
print(f"Closest match: Vector index {closest_match}, Similarity: {highest_similarity}")
print(f"Vectors in community: {community_vectors}")
