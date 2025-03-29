from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import re
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Request/Response Schemas
class QueryRequest(BaseModel):
    query: str
    history: list = []

class QueryResponse(BaseModel):
    response: str

# Global constants
INDEX_PATH = "faiss_openai_index"
DOC_PATHS = ["demo.pdf"]  # Replace with your actual PDF or TXT files
db = None
retriever = None

# Helper to remove <think> blocks
def remove_think_content(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Memory filter to strip <think> blocks
class FilteredConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        fi = {k: remove_think_content(v) if isinstance(v, str) else v for k, v in inputs.items()}
        fo = {k: remove_think_content(v) if isinstance(v, str) else v for k, v in outputs.items()}
        super().save_context(fi, fo)

# FAISS loader or builder
def load_faiss_openai():
    global db, retriever
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        return

    docs = []
    for path in DOC_PATHS:
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs.extend(splitter.split_documents(loaded_docs))

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    db.save_local(INDEX_PATH)

# Startup hook
@app.on_event("startup")
def startup_event():
    load_faiss_openai()

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
def query_handler(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="FAISS retriever not initialized")

    system_prompt = SystemMessagePromptTemplate.from_template("""
        You are an e-commerce helper bot designed to assist customers with all queries related to our featured product. Your tone must always be polite, friendly, and professional. Follow these guidelines at all times:
        Context-Driven Responses: Provide answers strictly based on the product context available. Do not deviate from the information provided in the knowledge base.
        Politeness and Courtesy: Remain courteous and respectful in every interaction. Never be rude or dismissive.
        Positive Comparisons: Proactively mention how our product excels compared to competitors using factual, context-based comparisons from our knowledge base. Always highlight the strengths of our product without disparaging competitors.
        Product Integrity: Never degrade or belittle the product you are selling. Emphasize its unique features, quality, and the value it brings to the customer.
        Customer Engagement: Create a welcoming and engaging atmosphere. Make customers feel involved in the purchase process and assure them that their well-being and satisfaction are our top priorities.
        Proactive Assistance: Anticipate customer needs and provide helpful, informative responses that guide them through their purchase decisions.
        Your goal is to ensure that every customer feels valued and confident in choosing our product, recognizing that it stands out in quality, service, and customer care compared to other options in the market.
""")
    human_prompt = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Question: {question}
    """)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

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
