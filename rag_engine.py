# rag_engine.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.llms import Ollama  # or your preferred chat wrapper
import os 

# 1. Text splitter: split into ~500‑char chunks with 100 overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 2. Embedding model: any HuggingFace‑based sentence embedder
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. LLM for generation
llm = Ollama(model="neural-chat")

def build_vectorstore(text: str,
                      persist_directory=".chromadb_rag",
                      collection_name="rag_collection",
                      update=True) -> Chroma:

    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    # Reuse existing DB if update is False
    if not update and os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )

    # Else build new (or overwrite)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vectordb.persist()
    return vectordb

def ask_with_rag(query: str, vectordb: Chroma, k: int = 4) -> str:
    results = vectordb.similarity_search(query, k=k)
    
    # Debug log
    print("🔍 Retrieved RAG chunks:")
    for doc in results:
        print("—", doc.page_content[:150])

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""Use ONLY the following context to answer the question.
If the answer is not in the context, say “I don’t know.”

Context:
\"\"\"{context}\"\"\"

Question: {query}
Answer:"""

    return llm.invoke(prompt)
