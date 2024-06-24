from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import Document
import openai
import os
import time
import pdfplumber
from dotenv import load_dotenv
load_dotenv()

openai.api_key=os.environ['OPENAI_API_KEY']

CHROMA_PATH="chroma"
DATA_PATH="data/Demian.pdf"


def load_pdf(pdf_path:str,start_page: int=0,top_margin=0.0,bottom_margin=0.0):
    print("load_pdf 작업중..")
    start_time =time.time()
    # 로더 정의
    loader = PDFPlumberLoader(file_path=pdf_path)
    # 파일 읽기
    data = loader.load()


    with pdfplumber.open(pdf_path) as pdf:
        for i,page in enumerate(pdf.pages):
            # Get the dimensions of the page
            width = page.width
            height = page.height
            if i>=start_page:
                crob_box = (0,top_margin,width,height-bottom_margin)
                cropped_page = page.within_bbox(crob_box)
                page_text= cropped_page.extract_text()
                data[i].page_content = page_text
  
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"load_pdf completed.\n[Duration of Time]: {duration_time:.2f} s")
    
    # 정의한 시작 페이지 부터 자르기
    data = data[start_page:]
    
    return data
  

# text 분할 함수
def make_chunks(documents: list[Document]):
    print("make_chunks 작업중..")
    start_time =time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents=documents)
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"Text Split to {len(chunks)} chunks.\n[Duration of Time]: {duration_time:.2f} s")

    return chunks

# chromadb에 저장
def save_to_db(chunks: list[Document]):
    print("save_to_db 작업중..")
    start_time =time.time()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace=embeddings.model,
    )

    # Create a new DB from the documents
    db = Chroma.from_documents(documents=chunks, embedding=cached_embedder, persist_directory=CHROMA_PATH)
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"Created and saved {len(chunks)} chunks to {CHROMA_PATH}.\n[Duration of Time]: {duration_time:.2f} s")

def main():
    generate_data_store()

# 파일 로드 및 데이터 저장
def generate_data_store():
    print("시작")
    # 파일 로드
    datas=load_pdf(pdf_path=DATA_PATH,top_margin=50,bottom_margin=50)

    # 텍스트 분활
    chunks = make_chunks(documents=datas)

    # chromadb에 저장
    save_to_db(chunks=chunks)


if __name__ == "__main__":
    main()