from gensim.models import Word2Vec

def train_word2vec(sentences):
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    return model

sentences = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"]
]

model = train_word2vec(sentences)
print(model.wv["learning"])
