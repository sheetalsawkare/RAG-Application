from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#document loader
data = PyPDFLoader("document_loaders/AI_in_Healthcare_Guide.pdf")
docs = data.load()

#text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
 )

chunks = splitter.split_documents(docs)

#create embeddings
embedding_model = MistralAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)
