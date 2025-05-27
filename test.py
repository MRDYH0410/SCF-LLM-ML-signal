from src.embedder import SentenceEmbedder

embedder = SentenceEmbedder()
vectors = embedder.encode([
    "Apple Inc. signed a new strategic partnership with CATL.",
    "Elon Musk announced the new Gigafactory site in Texas."
])
print("嵌入 shape:", vectors.shape)