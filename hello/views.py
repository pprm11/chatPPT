import os
import sys
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from django.shortcuts import render

from .models import Greeting

# Create your views here.


def index(request):
    return render(request, "index.html")


chat_history = []
def db(request):
    query = request.body.question

    import warnings
    warnings.filterwarnings("ignore")

    os.environ["OPENAI_API_KEY"] = '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'


    loader = PyPDFLoader("condo rules and regulations.pdf")
    sample_doc = loader.load()
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def split_docs(documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs

    docs = split_docs(sample_doc)

    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model_name="ada")

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.2})
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=compression_retriever
    )

    result = chain({
                       "question": "Role: victorian  Butler. \nAnswer based on the context and quote the rules. \nWhen you should write context write rules. \nIf you can't answer, ask for more explanation. \nQuestion: {}".format(
                           query), "chat_history": chat_history})
    print("Sir Sebastian Livviestone:")
    print(result['answer'])

    chat_history.append((query, result['answer']))

    return render(request, "db.html", {"Sir Sebastian Livviestone": result['answer']})
