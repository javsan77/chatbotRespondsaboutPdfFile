#Libraries
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings  # Modelo local, no requiere API
from langchain.vectorstores import Chroma
from langchain.llms import Ollama  # LLM local (ej: Llama 2)
from fastapi import FastAPI
import uvicorn

# 1. Cargar y vectorizar documentos (PDF)
loader = PyPDFLoader("manual.pdf")  # Reemplaza con tu archivo
documents = loader.load_and_split()

# 2. Crear base de datos vectorial local (Chroma)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# 3. Inicializar LLM local (Ollama)
llm = Ollama(model="llama2")  # Debes instalar Ollama primero (https://ollama.ai/)

# 4. Crear API REST con FastAPI
app = FastAPI()

@app.get("/chat")
def chat(query: str):
    # Buscar documentos similares
    docs = vectorstore.similarity_search(query, k=2)
    # Generar respuesta con el LLM
    respuesta = llm.invoke(f"Responde esta pregunta: {query}. Contexto: {docs[0].page_content}")
    return {"respuesta": respuesta}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    

