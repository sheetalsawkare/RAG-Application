from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k"  : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatMistralAI(model="mistral-small-2603")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful ai assistant"),
        ("human", "context: {context}  question: {question}")
    ]
)

print("rag system created")
print("press 0 to exit")

while True:
    query = input("YOU: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt_template.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")