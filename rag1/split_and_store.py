from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.embeddings import CacheBackedEmbeddings
import openai
import os
import time
import pdfplumber
from dotenv import load_dotenv
load_dotenv()

openai.api_key=os.environ['OPENAI_API_KEY']

CHROMA_PATH="chroma"
DATA_PATH="data/Demian.pdf"

# pdf 문서 로딩 함수
# 페이지 번호가 담긴 text를 반환한다.
def load_documents(pdf_path:str, start_page: int=0,crob_height=None):
    start_time =time.time()
    text=""
    with pdfplumber.open(pdf_path) as pdf:
        for i ,page in enumerate(pdf.pages):
            # Get the dimensions of the page
            width = page.width
            height = page.height
            
            if i>=start_page:
                if crob_height:
                    crob_box = (0,0,width,height-crob_height)
                    cropped_page = page.within_bbox(crob_box)
                    page_text = cropped_page.extract_text()+f"\n[Page {cropped_page.page_number}]\n"
                else:
                    page_text = page.extract_text()+f"\n[Page {cropped_page.page_number}]\n"
                text += page_text
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"Load text completed.\n[Duration of Time]: {duration_time:.2f} s")
    return text

# text 분할 함수
def make_chunk(text: str):
    start_time =time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_text(text=text)
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"Text Split to {len(chunks)} chunks.\n[Duration of Time]: {duration_time:.2f} s")

    return chunks

# chromadb에 저장
def save_to_db(chunks: list[str]):
    embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace=embeddings.model,
    )

    start_time =time.time()
    # Create a new DB from the documents
    db = Chroma.from_texts(texts=chunks, embedding=cached_embedder, persist_directory=CHROMA_PATH)
    end_time = time.time()
    duration_time=end_time-start_time
    print(f"Created and saved {len(chunks)} chunks to {CHROMA_PATH}.\n[Duration of Time]: {duration_time:.2f} s")

def main():
    generate_data_store()

# 파일 로드 및 데이터 저장
def generate_data_store():
    print("시작")
    # 파일 로드
    documents=load_documents(pdf_path=DATA_PATH,crob_height=50)
    # 텍스트 분활
    chunks = make_chunk(documents)
    # chromadb에 저장
    save_to_db(chunks=chunks)


if __name__ == "__main__":
    main()