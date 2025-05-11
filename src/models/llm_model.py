import requests
import json

def query_llm(
    prompt='What is the capital of France?',
    model_name='hermes-3-llama-3.2-3b',
    system_message='You are a helpful assistant.',
    temperature=0.7,
    max_tokens=100,
    stream=False,
    url='http://localhost:1234/v1/chat/completions'
    ):
    """
    Query a LLM running in Azure Databricks via POST API call.

    Args:
        prompt (str): The user prompt to send to the model.
        model_name (str): The name of the model loaded in LM Studio.
        system_message (str): System message to set context for the chat.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate.
        stream (bool): Whether to stream the response.
        url (str): The API endpoint URL.

    Returns:
        Response object: The response from the LLM server if successful.
        None: If the request failed.
    """
    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': stream
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error querying local LLM: {e}")
        return None
    
if __name__ == "__main__":
    # Example usage
    response = query_llm()
    if response:
        print("Response from LLM:")
        print(response.json())
    else:
        print("Failed to get a response from the LLM.")

