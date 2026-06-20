from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)
embedding = model.encode(
    "What is morality?"
)

print(type(embedding))
print(len(embedding))
print(embedding[:10])