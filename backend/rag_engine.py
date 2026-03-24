import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD PDF
# PyMuPDF reads the PDF and returns one Document object per page.
# Each Document has: .page_content (text) + .metadata (page number, filename)
# ─────────────────────────────────────────────────────────────
def load_document(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# ─────────────────────────────────────────────────────────────
# STEP 2: CHUNK
# We split documents into smaller pieces because:
# 1. LLMs have a context window limit
# 2. Smaller chunks = more precise retrieval
# 3. Overlap ensures answers spanning chunk boundaries aren't lost
# ─────────────────────────────────────────────────────────────
def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # ~700 characters per chunk (roughly 1-2 paragraphs)
        chunk_overlap=120,     # 120 char overlap between consecutive chunks
        separators=["\n\n", "\n", ". ", " "]  # split at natural boundaries first
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3: EMBED + INDEX
# text-embedding-3-small converts each chunk to a 1536-dimension vector.
# FAISS stores all vectors and enables lightning-fast similarity search.
# ─────────────────────────────────────────────────────────────
def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 4: BUILD QA CHAIN
# This connects: question → FAISS retrieval → GPT answer
# The prompt explicitly tells the model to stay grounded in the docs.
# ─────────────────────────────────────────────────────────────
def build_qa_chain(vector_store):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0   # 0 = deterministic, factual responses
    )

    prompt_template = """You are a precise Product Documentation Assistant.
Your job is to answer questions based ONLY on the provided product documentation context.

Rules:
- Answer only from the context provided. Do NOT use outside knowledge.
- If the answer isn't in the context, say: "This information isn't covered in the uploaded documentation. Please refer to the full docs or contact support."
- Be concise and structured. Use bullet points for multi-step answers.
- Mention the relevant section or topic if identifiable.

Context from documentation:
{context}

User Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE: PDF path → ready QA chain
# ─────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str):
    docs = load_document(pdf_path)
    chunks = split_into_chunks(docs)
    vector_store = build_vector_store(chunks)
    qa_chain = build_qa_chain(vector_store)
    return qa_chain, len(chunks), len(docs)


# ─────────────────────────────────────────────────────────────
# QUERY: question → answer + source citations
# ─────────────────────────────────────────────────────────────
def ask_question(qa_chain, question: str):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]

    sources = []
    seen_pages = set()
    for doc in source_docs:
        page_num = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content[:220].replace("\n", " ").strip()
        if page_num not in seen_pages:
            sources.append({"page": page_num, "snippet": snippet})
            seen_pages.add(page_num)

    return answer, sources
