import numpy as np
import pandas as pd
from vectorize import encode
from namh import find_closest_match
from tqdm import tqdm

# load data
data = pd.read_csv('data_vector.csv')

# convert embedding vectors to numpy arrays
embedding_list = [np.array(eval(embedding)).flatten() for embedding in data['embedding']]

# query similarity
input_text = "Clean makeup."

# encode the input text
print("Encoding input text...")
prompt_vector = encode(input_text)
prompt_vector = np.array(prompt_vector).flatten()

# check embedding_list sizing
for vec in embedding_list:
    assert len(vec) == len(prompt_vector), f"Vector dimensions mismatch: {len(vec)} != {len(prompt_vector)}"

# find_closest_match function
def find_closest_match_with_progress(prompt_vector, embedding_list):
    closest_match, highest_similarity, community_vectors = None, None, None
    for i, vec in enumerate(tqdm(embedding_list, desc="Finding closest match")):
        match, similarity, vectors = find_closest_match(prompt_vector, [vec])
        if highest_similarity is None or similarity > highest_similarity:
            closest_match, highest_similarity, community_vectors = match, similarity, vectors
    return closest_match, highest_similarity, community_vectors

# run the actual code with progress monitoring
closest_match, highest_similarity, community_vectors = find_closest_match_with_progress(prompt_vector, embedding_list)
print(f"Closest match: Vector index {closest_match}, Similarity: {highest_similarity}")
print(f"Vectors in community: {community_vectors}")
