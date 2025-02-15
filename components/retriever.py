from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
#from components.prompts import document_content_description, metadata_field_info
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = OllamaEmbeddings(model="jeffh/intfloat-multilingual-e5-large-instruct:f16")

llm = ChatOllama(model="qwen2.5:14b", temperature=0.8, num_ctx=12000)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

from uuid import uuid4

from langchain_core.documents import Document
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
uuids = [str(uuid4()) for _ in range(len(docs))]

#vector_store.add_documents(documents=docs, ids=uuids)
template = """ only answer the question, ignore the rest of the context that is not relevant to the question:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = vector_store.as_retriever(serach_type="similarity_search", )
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)
chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
#response=chain.invoke("suggest me some films with a rating lower then 8.0?")
#print(response)
#query = "suggest me a film with a higher rating then 8.3"
#docs = vector_store.similarity_search_with_score(query)
#print(docs)

response = llm.invoke("suggest me a film with a higher rating then 8.3")
print(response)