import openai
import numpy as np
import fitz  # PyMuPDF
import os
import time
from typing import List, Dict, Tuple
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema

# Set your OpenAI API key
openai.api_key = ""
client = openai.OpenAI(api_key=openai.api_key)

# Connect to ZillizDB
def connect_to_zillizdb() -> None:
    if not connections.has_connection("default"):
        print("Initializing ZillizDB (Milvus) client...")
        connections.connect("default", uri="", token="")
        print("ZillizDB (Milvus) client initialized successfully.")
    else:
        print("ZillizDB (Milvus) client already connected.")

connect_to_zillizdb()

# Create collection
def create_collection(collection_name: str, dim: int = 3072) -> Collection:
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id_field, embedding_field], description="Earnings reports embeddings")
    collection = Collection(name=collection_name, schema=schema)
    return collection

collection_name = "earnings_reports"
collection = create_collection(collection_name)

# Text extraction from PDFs
def extract_text_from_pdf(file_path: str) -> List[str]:
    document = fitz.open(file_path)
    chunks: List[str] = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text("text")
        chunks.append(page_text)
    return chunks

def extract_text_from_pdfs(folder_path: str) -> Dict[str, List[str]]:
    text_data: Dict[str, List[str]] = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            text_data[file_name] = extract_text_from_pdf(file_path)
    return text_data

folder_path = "earnings_pdfs"
pdf_texts = extract_text_from_pdfs(folder_path)

# Semantic chunking with table handling
def semantic_chunking(text: str) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    for paragraph in paragraphs:
        if is_table(paragraph):
            chunks.append(paragraph)  # Keep table as a single chunk
        else:
            chunks.append(paragraph.replace("\n", " "))
    return chunks

def is_table(paragraph: str) -> bool:
    # Simple heuristic to identify tables (e.g., lines with multiple delimiters or repeated whitespace)
    lines = paragraph.split('\n')
    table_like = all('\t' in line or '  ' in line for line in lines)
    return table_like

# Applying semantic chunking to PDFs
chunked_texts: Dict[str, List[str]] = {}
for file_name, pages in pdf_texts.items():
    all_chunks = []
    for page in pages:
        chunks = semantic_chunking(page)
        all_chunks.extend(chunks)
    chunked_texts[file_name] = all_chunks

# Function to get embedding
def get_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = np.array(response.data[0].embedding)
    return embedding

# Generate embeddings and IDs
embeddings: List[np.ndarray] = []
ids: List[int] = []
id_to_file_page: Dict[int, str] = {}
current_id = 1

for file_name, chunks in chunked_texts.items():
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        id_to_file_page[current_id] = f"{file_name}_chunk_{i}"
        ids.append(current_id)
        current_id += 1

# Save embeddings to a file (optional, to avoid recomputing)
np.save("embeddings.npy", embeddings)
np.save("ids.npy", ids)
np.save("id_to_file_page.npy", id_to_file_page)

# Data insertion into ZillizDB (with Index Creation)
def insert_embeddings(collection: Collection, ids: List[int], embeddings: List[np.ndarray]):
    entities = [
        ids,
        embeddings
    ]
    t0 = time.time()
    print("Inserting embeddings into ZillizDB...")
    collection.insert(entities)
    collection.flush()
    t1 = time.time()
    print(f"Embeddings inserted successfully into ZillizDB in {round(t1 - t0, 4)} seconds.")

def create_index(collection: Collection):
    print("Creating index for the collection...")
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    t0 = time.time()
    collection.load()
    t1 = time.time()
    print(f"Index created and collection loaded successfully in {round(t1 - t0, 4)} seconds.")

def load_collection(collection: Collection) -> None:
    print("Loading collection into memory...")
    t0 = time.time()
    collection.load()
    t1 = time.time()
    print(f"Collection loaded successfully in {round(t1 - t0, 4)} seconds.")

insert_embeddings(collection, ids, embeddings)
create_index(collection)

# Search and query LLM
def search_embeddings(collection: Collection, query_embedding: np.ndarray, top_k: int = 5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    t0 = time.time()
    print("Searching embeddings in ZillizDB...")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["id"]
    )
    t1 = time.time()
    print(f"Search in ZillizDB completed successfully in {round(t1 - t0, 4)} seconds.")
    return results

# Enhance prompt
def enhance_prompt(query: str, search_results, pdf_texts: Dict[str, List[str]], id_to_file_page: Dict[int, str]) -> str:
    enhanced_prompt = query + "\n\nHere is some relevant information from the documents:\n"
    for result in search_results:
        for hit in result:
            file_page = id_to_file_page[hit.entity.get("id")]
            file_name, chunk_num = file_page.rsplit('_chunk_', 1)
            chunk_num = int(chunk_num)
            chunk_text = pdf_texts[file_name][chunk_num]
            enhanced_prompt += f"\nFrom {file_name}, chunk {chunk_num}:\n{chunk_text}\n"
    return enhanced_prompt

def format_for_chat_api(enhanced_prompt: str) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers questions about Google's earnings breakdowns."},
        {"role": "user", "content": enhanced_prompt}
    ]
    return messages

def query_llm(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500  # Adjust as needed
    )
    return response.choices[0].message.content.strip()

# Example query
