import requests
import json

def get_embedding(
    texts,
    model_name='text-embedding-nomic-embed-text-v1.5',
    task_type='search_document',
    dimensionality=256,
    url='http://localhost:1234/v1/embeddings',
):
    """
    Query local LM Studio for embeddings from nomic-embed-text-v1.5 model.

    Args:
        texts (list of str): List of texts to embed.
        model_name (str): Model name loaded in LM Studio.
        task_type (str): Task instruction prefix, e.g. 'search_document', 'search_query'.
        dimensionality (int): Optional embedding size (64, 128, 256, 512, 768).
        url (str): Local LM Studio embeddings endpoint.

    Returns:
        dict: JSON response containing embeddings if successful, else None.
    """
    headers = {
        'Content-Type': 'application/json'
    }

    # Prepare input texts with task prefix
    prefixed_texts = [f"{task_type}: {text}" for text in texts]

    payload = {
        "model": model_name,
        "input": prefixed_texts,
        "dimensionality": dimensionality
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        result_data = result['data']
        embedding_list = [item['embedding'] for item in result_data]
        # Return the first embedding if only one text is provided
        return embedding_list
    except requests.exceptions.RequestException as e:
        print(f"Error querying embeddings: {e}")
        return None


if __name__ == "__main__":
    texts_to_embed = [
        "What is GEN-AI?",

    ]

    embeddings_response = get_embedding(
        texts=texts_to_embed,
        task_type='search_query',  # or 'search_document', 'clustering', 'classification'
        dimensionality=256
    )

    if embeddings_response:
        print("Embeddings:", embeddings_response)
    else:
        print("Failed to get embeddings.")
