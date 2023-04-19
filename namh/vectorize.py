import torch
import transformers
import pandas as pd
from typing import List

# load pre-trained llm and tokenizer
model_name = 'bert-base-cased'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(model_name)

# define a function to encode text into a vector embedding
def encode(text: str) -> List[float]:
    # Truncate the text to fit within the maximum sequence length
    max_length = tokenizer.max_model_input_sizes[model_name]
    text = text[:max_length - 2]  # Subtract 2 for the [CLS] and [SEP] tokens

    # Encode the text and add the [CLS] and [SEP] tokens
    input_ids = torch.tensor(tokenizer.encode('[CLS]' + text + '[SEP]')).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state

    # Take the first token's embedding as the subject line's representation
    embedding = last_hidden_state[:, 0, :].detach().numpy()
    return embedding.tolist()

if __name__ == '__main__':
    # get data
    data = pd.read_csv('data.csv')

    # Encode the subject lines into vector embeddings
    embeddings = [encode(text) for text in data['description']]
    data['embedding'] = embeddings

    # Convert array of floats to dense vector
    data['embedding_vector'] = data['embedding'].apply(lambda x: torch.tensor(x).squeeze())
    
    # Print the resulting dataframe
    print(data)
    data.to_csv('data_vector.csv')
