from langchain_community.document_loaders import TextLoader

data = TextLoader("document_loaders/ai_article.txt")
docs = data.load()
print(docs[0])