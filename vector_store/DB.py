from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

documents = [
    Document(page_content="""Artificial Intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that typically require  human intelligence, such as learning, reasoning, and decision-making.""",metadata={"source": "ai.txt", "topic": "AI"} ),
    Document( page_content=""" Cloud Computing provides on-demand access to computing resources such as servers, storage, databases, and networking over the internet. It helps organizations scale applications efficiently. """, metadata={"source": "cloud.txt", "topic": "Cloud Computing"} ),
    Document( page_content=""" Machine Learning is a subset of Artificial Intelligence that enables systems to learn patterns from data and improve performance without being explicitly programmed. """, metadata={"source": "ml.txt", "topic": "Machine Learning"})
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is machine learning?",k=2)

for r in result:
    print(r)