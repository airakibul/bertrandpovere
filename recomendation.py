from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 1. Load PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

extracted_data = load_pdf_file(data="data/")

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(extracted_data)

# 3. Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Build RetrievalQA chain
llm = ChatOpenAI(model="gpt-4o-mini")  # or gpt-3.5-turbo
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Ask a question
query = "What are the main findings of the document?"
result = qa_chain.invoke(query)

print("Answer:", result["result"])
print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])
