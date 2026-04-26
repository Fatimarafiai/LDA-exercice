import os
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
from data_loader import load_documents
from config import LDA_CONFIG

def main():
    print("="*60)
    print("LDA (Latent Dirichlet Allocation) - Advanced Implementation")
    print("="*60)
    
    # Load documents
    print("\n[1/5] Loading documents...")
    documents = load_documents('data/sample_documents.txt')
    print(f"Loaded {len(documents)} documents")
    
    # Vectorize
    print("\n[2/5] Vectorizing documents...")
    vectorizer = CountVectorizer(
        max_df=LDA_CONFIG['max_df'],
        min_df=LDA_CONFIG['min_df'],
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(feature_names)} words")
    
    # Train LDA
    print("\n[3/5] Training LDA model...")
    lda = LatentDirichletAllocation(
        n_components=LDA_CONFIG['n_topics'],
        random_state=LDA_CONFIG['random_state'],
        max_iter=LDA_CONFIG['max_iter'],
        learning_method='online',
        n_jobs=-1
    )
    lda.fit(doc_term_matrix)
    print(f"Model trained with {LDA_CONFIG['n_topics']} topics")
    
    # Display topics
    print("\n[4/5] Topics discovered:")
    print("="*60)
    display_topics(lda, feature_names, LDA_CONFIG['n_words'])
    
    # Evaluate
    print("\n[5/5] Model Evaluation:")
    print("="*60)
    perplexity = lda.perplexity(doc_term_matrix)
    print(f"Perplexity: {perplexity:.4f}")
    
    # Document-topic distribution
    doc_topic_dist = lda.transform(doc_term_matrix)
    print(f"\nDocument-Topic Matrix shape: {doc_topic_dist.shape}")
    print(f"Average topic distribution per document:")
    print(np.mean(doc_topic_dist, axis=0))
    
    print("\n" + "="*60)
    print("LDA Analysis Complete!")
    print("="*60)

def display_topics(lda, feature_names, n_words=10):
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        print(f"\nTopic #{topic_idx + 1}:")
        for word, weight in zip(top_words, top_weights):
            print(f"  - {word}: {weight:.4f}")

if __name__ == "__main__":
    main()
