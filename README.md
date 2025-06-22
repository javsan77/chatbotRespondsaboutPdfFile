````markdown
# 📚 PDF Chatbot en Español usando LangChain, Chroma, HuggingFace y Ollama

Este proyecto construye una API REST con FastAPI que permite realizar preguntas en español sobre el contenido de un archivo PDF. Utiliza técnicas modernas de NLP y recuperación de información como **LangChain**, **Chroma**, **HuggingFace Embeddings** y un modelo LLM local con **Ollama** (ej. LLaMA 2).

## 🚀 Características

- Carga y divide archivos PDF con LangChain.
- Genera y persiste una base vectorial con ChromaDB.
- Usa embeddings multilingües para recuperar documentos relevantes.
- Integra un modelo local como LLaMA 2 vía Ollama para generar respuestas en español.
- Expone una API REST para consultas.

---

## 📦 Requisitos

- Python 3.10+
- [Ollama instalado](https://ollama.ai)
- Dependencias:

```bash
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama fastapi uvicorn
````

---

## 🧠 Modelos y Archivos

* 📄 `manual.pdf`: documento fuente para preguntas (coloca el tuyo en el mismo directorio).
* 🧠 Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* 💬 LLM: `llama2` (puede ser otro compatible con Ollama).

---

## 🛠️ Estructura del Código

### 1. Indexación y Embeddings

```python
loader = PyPDFLoader("manual.pdf")
documents = loader.load_and_split()
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
```

Si ya existe una base vectorial, esta se carga automáticamente.

### 2. Inicialización del LLM

```python
llm = OllamaLLM(model="llama2")
```

Asegúrate de tener el modelo descargado en tu entorno Ollama.

### 3. API REST

```python
@app.get("/chat")
def chat(query: str):
    docs = vectorstore.similarity_search(query, k=2)
    context_text = docs[0].page_content
    prompt = f"..."
    respuesta = llm.invoke(prompt)
    return {"respuesta": respuesta}
```

---

## ▶️ Ejecución

```bash
python app.py
```

La API se ejecutará en: `http://localhost:8000/chat?query=tu_pregunta`

---

## 📤 Ejemplo de Uso

```bash
curl "http://localhost:8000/chat?query=¿Cuál es el objetivo del manual?"
```

Respuesta:

```json
{
  "respuesta": "El objetivo del manual es..."
}
```

---

## 📁 Directorios Generados

* `./chroma_db/`: contiene la base de datos vectorial persistente (`chroma.sqlite3`, etc).

---

## 🧪 Recomendaciones

* Usa PDFs en español para obtener respuestas más coherentes.
* El modelo `llama2` debe estar previamente descargado con:

```bash
ollama run llama2
```

---

## 📄 Licencia

MIT

---

## ✨ Autor

Desarrollado por [Tu Nombre o Usuario](https://github.com/tuusuario)

```

---

¿Te gustaría que también genere el `requirements.txt` o un `Dockerfile` para desplegarlo fácilmente?
```
