# RAG with Model Context Protocol (MCP)

This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex with Model Context Protocol (MCP) integration.

## What is Model Context Protocol (MCP)?

Model Context Protocol (MCP) is a framework for structuring and formatting context that's provided to large language models (LLMs). It helps improve the quality and relevance of LLM responses by providing a consistent way to format and prioritize context information.

Key benefits of MCP:

1. **Structured Context**: Organizes retrieved information in a consistent format
2. **Source Attribution**: Clearly identifies the source of each piece of information
3. **Relevance Indicators**: Signals to the model which information is most relevant
4. **Improved Response Quality**: Helps the model generate more accurate and relevant responses

## Project Structure

```
rag/
├── data/
│   ├── file_uploads/     # Directory for uploaded files
│   └── url_uploads/      # Directory for URL content
├── src/
│   ├── chat-ui.py        # Original chat UI implementation
│   ├── rag_with_mcp.py   # RAG with MCP implementation
│   └── integrate_mcp_with_chat.py  # Integration of RAG+MCP with chat UI
├── storage/              # Directory for vector index storage
├── .env.example          # Example environment variables
└── requirements.txt      # Project dependencies
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```
   cp .env.example .env
   ```
4. Edit `.env` and add your OpenAI API key

## Usage

Run the chat interface with RAG+MCP:

```
python src/integrate_mcp_with_chat.py
```

This will start a Gradio web interface where you can:
- Upload documents (PDF, DOCX, TXT, etc.)
- Add URLs to include web content
- Ask questions about your documents

## How It Works

1. **Document Processing**: Documents are uploaded and processed into chunks
2. **Indexing**: Document chunks are embedded and stored in a vector index
3. **Retrieval**: When a query is received, relevant chunks are retrieved
4. **MCP Formatting**: Retrieved chunks are formatted according to MCP guidelines
5. **Response Generation**: The LLM generates a response based on the formatted context

## MCP Implementation Details

Our implementation formats context as follows:

```
# CONTEXT

## Source 1: document1.pdf (page 5) | Relevance: ★★★

Content from the most relevant chunk...

## Source 2: document2.txt | Relevance: ★★★

Content from the second most relevant chunk...

...
```

This structured format helps the LLM understand:
- The source of each piece of information
- The relative importance of different chunks
- The hierarchical organization of the context

## Customization

You can customize the RAG pipeline and MCP implementation by modifying:

- `rag_with_mcp.py`: Change embedding models, chunk sizes, or MCP formatting
- `integrate_mcp_with_chat.py`: Modify the chat UI or response generation

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt