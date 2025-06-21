#Libraries
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings  # Modelo local, no requiere API
from langchain.vectorstores import Chroma
from langchain.llms import Ollama  # LLM local (ej: Llama 2)
from fastapi import FastAPI
import uvicorn
