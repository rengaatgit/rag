import requests

def query_llm(prompt, model_id="hermes-3-llama-3.2-3b"):
    url = "http://localhost:1236/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio"  # default API key if enabled
    }
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Example usage
llm_response = query_llm("Explain reinforcement learning.")
print("LLM response:", llm_response)
