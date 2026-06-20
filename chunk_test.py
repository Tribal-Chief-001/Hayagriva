from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(
    "data/documents/bc_Arendt_Kant_CritiquePracticalReason.pdf"
)

pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(pages)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0].page_content)