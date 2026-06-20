from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/documents/bc_Arendt_Kant_CritiquePracticalReason.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

print(f"Pages Loaded: {len(pages)}")

print("\nFIRST PAGE:\n")
print(pages[0].page_content[:1000])