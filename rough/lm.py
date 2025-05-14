import requests

def get_embedding(text, model_id="text-embedding-nomic-embed-text-v1.5"):
    url = "http://localhost:1236/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio"
    }
    data = {
        "model": model_id,
        "input": text
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    result = response.json()
    # The embedding vector is usually in result["data"][0]["embedding"]
    return result["data"][0]["embedding"]

# Example usage
embedding_vector = get_embedding("Machine learning is fascinating.")
print("Embedding vector length:", len(embedding_vector))
