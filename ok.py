import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import json

TEXT_FILE_PATH = "data.txt"
VECTOR_DB_PATH = "vectorstores/db_faiss"
MODEL_PATH = "models/all-MiniLM-L6-v2-f16.gguf"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

def create_db_from_text():
    if not os.path.exists(TEXT_FILE_PATH):
        raise ValueError(f"Không tìm thấy file {TEXT_FILE_PATH}")

    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    jsonList = json.loads(raw_text)
    dataList = [json.dumps(item, ensure_ascii=False) for item in jsonList]

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
    # chunks = text_splitter.split_text(raw_text)

    embedding_model = GPT4AllEmbeddings(model_file=MODEL_PATH)

    db = FAISS.from_texts(texts=dataList, embedding=embedding_model)
    db.save_local(VECTOR_DB_PATH)

    print(f"Đã tạo FAISS DB và lưu")

@app.post("/search")
def search_vectorstore(query: SearchQuery):
    if not os.path.exists(VECTOR_DB_PATH):
        raise HTTPException(status_code=500, detail="FAISS DB chưa được tạo!")

    # Load FAISS DB
    embedding_model = GPT4AllEmbeddings(model_file=MODEL_PATH)
    db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

    # tìm kiếm
    results = db.similarity_search(query.query, k=query.top_k)
    print(query.dict())

    return {"results": [doc.page_content for doc in results]}

if __name__ == "__main__":
    create_db_from_text()
    import uvicorn
    print("🚀 FastAPI Server đang chạy tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
