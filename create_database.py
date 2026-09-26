import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)


# ============================================================
# 1. PDF PATH
# ============================================================

PDF_PATH = "NIPS-2017-attention-is-all-you-need-Paper.pdf"


# ============================================================
# 2. LOAD PDF
# ============================================================

print("\nLoading PDF...")

loader = PyPDFLoader(PDF_PATH)

pages = loader.load()

print(f"Loaded {len(pages)} pages.")


# ============================================================
# 3. SPLIT DOCUMENT
# ============================================================

print("\nSplitting document...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

chunks = splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks.")


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# ============================================================
# 5. CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="./Chromadb"
)


# ============================================================
# 6. DELETE OLD COLLECTION
# ============================================================

try:

    client.delete_collection(
        name="Data"
    )

    print("\nOld Data collection deleted.")

except Exception:

    print("\nNo old Data collection found.")


# ============================================================
# 7. CREATE COLLECTION
# ============================================================

collection = client.create_collection(
    name="Data",
    embedding_function=embedding_function
)

print("New Data collection created.")


# ============================================================
# 8. PREPARE DOCUMENTS
# ============================================================

documents = [
    chunk.page_content
    for chunk in chunks
]

ids = [
    f"doc_{i}"
    for i in range(len(documents))
]


# ============================================================
# 9. ADD DOCUMENTS TO CHROMADB
# ============================================================

print("\nAdding documents to ChromaDB...")

collection.add(
    documents=documents,
    ids=ids
)


# ============================================================
# 10. VERIFY DATABASE
# ============================================================

print("\n" + "=" * 60)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"Collection name : {collection.name}"
)

print(
    f"Documents stored: {collection.count()}"
)

print("=" * 60)