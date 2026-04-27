import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_loader import load_documents
from config import LDA_CONFIG

def plot_topics(lda, vectorizer, num_words=10):
    """Affiche les top words de chaque topic"""
    feature_names = vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        axes[topic_idx].barh(range(len(top_words)), top_weights, color='steelblue')
        axes[topic_idx].set_yticks(range(len(top_words)))
        axes[topic_idx].set_yticklabels(top_words)
        axes[topic_idx].set_xlabel('Weight')
        axes[topic_idx].set_title(f'Topic #{topic_idx + 1}')
        axes[topic_idx].invert_yaxis()
    
    plt.suptitle('Top Words by Topic', fontsize=16, fontweight='bold')
    return fig

def plot_heatmap(lda, doc_term_matrix):
    """Affiche une heatmap de la distribution document-topic"""
    doc_topic_dist = lda.transform(doc_term_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(doc_topic_dist, cmap='YlOrRd', cbar=True, ax=ax)
    ax.set_xlabel('Topics')
    ax.set_ylabel('Documents')
    ax.set_title('Document-Topic Distribution Heatmap')
    
    return fig

def plot_word_clouds(lda, vectorizer, num_words=15):
    """Affiche les mots principaux pour chaque topic"""
    feature_names = vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        # Normaliser les poids pour la taille
        sizes = (top_weights - top_weights.min()) / (top_weights.max() - top_weights.min()) * 100 + 20
        
        axes[topic_idx].scatter(range(len(top_words)), top_weights, s=sizes, alpha=0.6, color='coral')
        for i, word in enumerate(top_words):
            axes[topic_idx].text(i, top_weights[i], word, ha='center', va='center', fontweight='bold')
        
        axes[topic_idx].set_xticks([])
        axes[topic_idx].set_ylabel('Weight')
        axes[topic_idx].set_title(f'Topic #{topic_idx + 1} Words')
    
    plt.suptitle('Word Clouds by Topic', fontsize=16, fontweight='bold')
    return fig

def plot_topic_distribution(lda, doc_term_matrix):
    """Affiche la distribution des topics"""
    doc_topic_dist = lda.transform(doc_term_matrix)
    mean_dist = np.mean(doc_topic_dist, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax.pie(mean_dist, labels=[f'Topic {i+1}' for i in range(len(mean_dist))], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Average Topic Distribution', fontsize=14, fontweight='bold')
    
    return fig

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("📊 LDA Visualizations")
    print("="*60)
    
    # Charger les données
    print("\n[1/4] Loading documents...")
    documents = load_documents('data/sample_documents.txt')
    print(f"✅ {len(documents)} documents loaded")
    
    # Vectorizer
    print("\n[2/4] Vectorizing documents...")
    vectorizer = CountVectorizer(
        max_df=LDA_CONFIG['max_df'],
        min_df=LDA_CONFIG['min_df'],
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(documents)
    print(f"✅ Vocabulary size: {len(vectorizer.get_feature_names_out())} words")
    
    # Entraîner LDA
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
    
    # Créer les visualisations
    print("\n[4/4] Creating visualizations...")
    
    # 1. Top words par topic
    fig1 = plot_topics(lda, vectorizer, num_words=10)
    plt.savefig('output/top_words_by_topic.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: top_words_by_topic.png")
    
    # 2. Heatmap
    fig2 = plot_heatmap(lda, doc_term_matrix)
    plt.savefig('output/document_topic_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: document_topic_heatmap.png")
    
    # 3. Word clouds
    fig3 = plot_word_clouds(lda, vectorizer, num_words=15)
    plt.savefig('output/word_clouds_by_topic.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: word_clouds_by_topic.png")
    
    # 4. Topic distribution
    fig4 = plot_topic_distribution(lda, doc_term_matrix)
    plt.savefig('output/topic_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: topic_distribution.png")
    
    print("\n" + "="*60)
    print("🎉 All visualizations are ready!")
    print("="*60)
    plt.show()
   
   
    
    
