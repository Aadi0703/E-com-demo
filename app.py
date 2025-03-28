from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import os
import re

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Request and response schemas
class QueryRequest(BaseModel):
    query: str
    history: list = []

class QueryResponse(BaseModel):
    response: str

# FAISS paths and global vars
INDEX_PATH = "faiss_ollama_index"
DOC_PATHS = ["demo.pdf"]  

db = None
retriever = None

# Filter <think> content
def remove_think_content(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Conversation memory filter
class FilteredConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        filtered_inputs = {k: remove_think_content(v) if isinstance(v, str) else v for k, v in inputs.items()}
        filtered_outputs = {k: remove_think_content(v) if isinstance(v, str) else v for k, v in outputs.items()}
        super().save_context(filtered_inputs, filtered_outputs)

# Load FAISS or build from docs
def load_faiss_with_ollama():
    global db, retriever
    embeddings = OllamaEmbeddings(model="llama3.1:8b")  # You can change model here

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        return

    all_docs = []
    for path in DOC_PATHS:
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_docs.extend(splitter.split_documents(docs))

    db = FAISS.from_documents(all_docs, embeddings)
    retriever = db.as_retriever()
    db.save_local(INDEX_PATH)

# Load FAISS at startup
@app.on_event("startup")
def startup_event():
    load_faiss_with_ollama()

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
def query_handler(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="FAISS retriever not loaded")

    system_prompt = SystemMessagePromptTemplate.from_template("""
        You are a helpful teaching assistant. Only use the provided context to answer.
        Don't use external or global knowledge. Filter out <think> blocks before replying.
    """)

    human_prompt = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Question: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    combined_input = " ".join([item.get("content", "") for item in request.history] + [request.query])
    result = qa_chain.invoke({"question": combined_input})
    answer = remove_think_content(result["answer"])

    return QueryResponse(response=answer)
