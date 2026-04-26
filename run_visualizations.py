import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_loader import load_documents
from config import LDA_CONFIG
from visualizations import plot_topics, plot_heatmap, plot_word_clouds, plot_topic_distribution

def main():
    print("="*60)
    print("📊 LDA Visualizations")
    print("="*60)
    
    # [1] Charger les données
    print("\n[1/4] Loading documents...")
    documents = load_documents('data/sample_documents.txt')
    print(f"✅ {len(documents)} documents loaded")
    
    # [2] Vectorizer
    print("\n[2/4] Vectorizing documents...")
    vectorizer = CountVectorizer(
        max_df=LDA_CONFIG['max_df'],
        min_df=LDA_CONFIG['min_df'],
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(documents)
    print(f"✅ Vocabulary size: {len(vectorizer.get_feature_names_out())} words")
    
    # [3] Entraîner LDA
    print("\n[3/4] Training LDA model...")
    lda = LatentDirichletAllocation(
        n_components=LDA_CONFIG['n_topics'],
        random_state=LDA_CONFIG['random_state'],
        max_iter=LDA_CONFIG['max_iter'],
        learning_method='online',
        n_jobs=-1
    )
    lda.fit(doc_term_matrix)
    print(f"✅ Model trained with {LDA_CONFIG['n_topics']} topics")
    
    # [4] Créer les visualisations
    print("\n[4/4] Creating visualizations...")
    
    fig1 = plot_topics(lda, vectorizer, num_words=10)
    plt.savefig('output/top_words_by_topic.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: top_words_by_topic.png")
    plt.close()
    
    fig2 = plot_heatmap(lda, doc_term_matrix)
    plt.savefig('output/document_topic_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: document_topic_heatmap.png")
    plt.close()
    
    fig3 = plot_word_clouds(lda, vectorizer, num_words=15)
    plt.savefig('output/word_clouds_by_topic.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: word_clouds_by_topic.png")
    plt.close()
    
    fig4 = plot_topic_distribution(lda, doc_term_matrix)
    plt.savefig('output/topic_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: topic_distribution.png")
    plt.close()
    
    print("\n" + "="*60)
    print("🎉 All visualizations are ready!")
    print("="*60)

if __name__ == "__main__":
    main()
