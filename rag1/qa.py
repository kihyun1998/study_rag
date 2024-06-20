import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema import Document
from langchain import hub
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
import os
load_dotenv()

openai.api_key=os.environ['OPENAI_API_KEY']

CHROMA_PATH="chroma"


def main():
    # 질문하기
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # db
    embeddings = OpenAIEmbeddings()
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace=embeddings.model,
    )

    db = Chroma(embedding_function=cached_embedder,persist_directory=CHROMA_PATH)

    # 검색기 생성
    retriever = db.as_retriever(search_kwargs={"k": 1})

    # 프롬프트 가져오기
    prompt = hub.pull("rlm/rag-prompt")

    # LLM 정의
    llm = ChatOpenAI(model='gpt-4o')

    def format_docs(docs:Document):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 체인 생성
    chain = (
        {"context": retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(chain.invoke(query_text))


if __name__ == "__main__":
    main()