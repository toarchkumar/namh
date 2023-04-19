import torch
import transformers
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm

# load pre-trained llm and tokenizer
def get_tokenizer_and_model(model_name = 'bert-base-cased'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModel.from_pretrained(model_name)
    return tokenizer, model

# define a function to encode text into a vector embedding
def encode(texts: List[str], model_name, tokenizer, model, batch_size=16) -> torch.Tensor:
    max_length = tokenizer.max_model_input_sizes[model_name]
    
    # truncate and encode texts
    encoded_texts = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = encoded_texts['input_ids']

    # create a DataLoader to handle batching
    dataset = torch.utils.data.TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # encode the texts in batches
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding texts"):
            outputs = model(*batch)
            last_hidden_state = outputs.last_hidden_state
            batch_embeddings = last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings)

    # stack the embeddings and return as a tensor
    return torch.cat(embeddings, dim=0)

if __name__ == '__main__':
    # get data
    data = pd.read_csv('data.csv')

    # get tokenizer, model
    model_name = 'bert-base-cased'
    tokenizer, model = get_tokenizer_and_model(model_name)

    # encode the subject lines into vector embeddings
    embeddings = encode(data['description'].tolist(), tokenizer, model)
    data['embedding_vector'] = embeddings.tolist()

    # print the result and save
    print(data)
    data.to_csv('data_vector.csv')
