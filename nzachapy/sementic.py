from sentence_transformers import SentenceTransformer, util
import httpx
from openai import OpenAI
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')


def embed(data, openai_api_key=None):
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key, http_client=httpx.Client(timeout=None))

        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=data
        )
        return response.data[0].embedding


    data_embedding = model.encode(data, convert_to_tensor=True).tolist()

    return data_embedding

def match_embeddings(data_embedding_list, query_embedding, threshold):
    data_embeddings = torch.stack([torch.tensor(c) for c in data_embedding_list])

    similarities = util.cos_sim(query_embedding, data_embeddings)[0]

    related = [
        {"index": i, "score": float(similarities[i])}
        for i in range(len(data_embedding_list))
        if similarities[i] >= threshold
    ]

    related.sort(key=lambda x: x["score"], reverse=True)

    return related