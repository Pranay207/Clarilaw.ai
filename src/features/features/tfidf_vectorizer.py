from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

texts = [
    "I love machine learning",
    "Machine learning is powerful"
]

X, vectorizer = build_tfidf(texts)
print(X.toarray())
