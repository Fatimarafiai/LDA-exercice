import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Exemple d'utilisation
documents = [
    "machine learning is fascinating",
    "data science and AI",
    "python programming language"
]

vectorizer = CountVectorizer(stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(doc_term_matrix)

print("Topics:", lda.components_)
