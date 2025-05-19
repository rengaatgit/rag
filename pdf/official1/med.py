#https://bhavikjikadara.medium.com/llamaindex-chroma-building-a-simple-rag-pipeline-cd67fc184190
# 
# # To install these libraries, you can run the following commands:
##pip install chromadb llama-index
import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

llm = Ollama(model="llama3")
response = llm.complete("Who is Laurie Voss? Write in 10 words")
print(response)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(input_files=["./resume.pdf"]).load_data()
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("ollama")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model,
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=10)]
)


template = (
    "Imagine you are a data scientist's assistant and "
    "you answer a recruiter's questions about the data scientist's experience."
    "Here is some context from the data scientist's "
    "resume related to the query::\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, "
    "Please respond to the following inquiry:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly and ensure your response is "
    "clear to someone without a data science background."
    "The data scientist's name is Bhavik Jikadara."
)
qa_template = PromptTemplate(template)


query_engine = index.as_query_engine(
    text_qa_template=qa_template,
    similarity_top_k=3
)


response = query_engine.query("Do you have experience with Python?")
print(response.response)

