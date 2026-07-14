from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/AI_in_Healthcare_Guide.pdf")

docs = data.load()

print(len(docs))