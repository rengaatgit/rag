from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="hermes-3-llama-3.2-3b",
    api_key="lm-studio",               # default LM Studio API key; adjust if needed
    base_url="http://localhost:1236/v1",  # your local LM Studio endpoint
)

# Example usage: generate a completion
response = llm.complete("Hello, LM Studio!")
print(response)
