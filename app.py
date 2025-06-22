# Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from fastapi import FastAPI
import uvicorn
import os # Import the os module

# Configuration for Chroma persistence
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PDF_FILE = "manual.pdf"

# Initialize embeddings once
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# 1. Check if the vectorstore already exists and load it, otherwise create it.
if os.path.exists(PERSIST_DIRECTORY) and os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
    print(f"Loading existing Chroma vectorstore from {PERSIST_DIRECTORY}...")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print("Vectorstore loaded.")
else:
    print(f"Creating and persisting new Chroma vectorstore in {PERSIST_DIRECTORY}...")
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load_and_split()
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIRECTORY)
    print("Vectorstore created and persisted.")

# 2. Inicializar LLM local (Ollama)
llm = OllamaLLM(model="llama2") # Debes instalar Ollama primero (https://ollama.ai/)

# 3. Crear API REST con FastAPI
app = FastAPI()

@app.get("/chat")
def chat(query: str):
    # Buscar documentos similares
    docs = vectorstore.similarity_search(query, k=2)

    # Prepare the context (assuming docs[0].page_content is already relevant text)
    context_text = docs[0].page_content
    
    # If the original PDF content (context_text) is in English, this might still be challenging
    # for the model to entirely translate on the fly while generating a good answer.
    prompt_spanish = (
        f"Responde la siguiente pregunta en español, utilizando el siguiente contexto. "
        f"Pregunta: {query}\n"
        f"Contexto: {context_text}\n"
        f"Respuesta en español:"
    )

        # Generar respuesta con el LLM
    respuesta = llm.invoke(prompt_spanish) # Use the updated prompt
    return {"respuesta": respuesta}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)