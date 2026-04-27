import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_loader import load_documents
from config import LDA_CONFIG

# ============================================
# 📊 FONCTION 1: Top Words - PAR TOPIC (Séparé)
# ============================================
def plot_top_words_by_topic(lda, vectorizer, num_words=12):
    """Crée UN graphique séparé pour chaque topic"""
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        # 🎨 Figure grande et claire
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 📊 Barh avec couleurs dégradées
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_words)))
        ax.barh(range(len(top_words)), top_weights, color=colors, edgecolor='black', linewidth=1.5)
        
        # 📝 Configuration de l'axe
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=13, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=14, fontweight='bold')
        ax.set_title(f'Topic #{topic_idx + 1} - Top {num_words} Words', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # 🎯 Ajouter les valeurs sur les barres
        for i, (word, weight) in enumerate(zip(top_words, top_weights)):
            ax.text(weight + 0.1, i, f'{weight:.2f}', 
                   va='center', fontsize=11, fontweight='bold')
        
        # 🔲 Grid pour meilleure lisibilité
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(f'output/topic_{topic_idx + 1}_top_words.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: topic_{topic_idx + 1}_top_words.png")
        plt.close()

# ============================================
# 🔥 FONCTION 2: Heatmap Améliorée
# ============================================
def plot_heatmap_improved(lda, doc_term_matrix):
    """Heatmap bien formatée et lisible"""
    doc_topic_dist = lda.transform(doc_term_matrix)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 🎨 Heatmap avec meilleure colormap
    sns.heatmap(doc_topic_dist, cmap='RdYlBu_r', cbar=True, ax=ax,
                cbar_kws={'label': 'Topic Probability'}, 
                vmin=0, vmax=1, linewidths=0.5)
    
    # 📝 Labels et titres
    ax.set_xlabel('Topics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Documents', fontsize=14, fontweight='bold')
    ax.set_title('Document-Topic Distribution Heatmap', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 📊 Améliorations
    ax.set_xticklabels([f'Topic {i+1}' for i in range(lda.n_components)], 
                       fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/document_topic_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: document_topic_heatmap.png")
    plt.close()

# ============================================
# 💫 FONCTION 3: Word Clouds Améliorés (Séparé)
# ============================================
def plot_word_clouds_improved(lda, vectorizer, num_words=15):
    """Crée UN graphique séparé pour chaque topic avec word cloud"""
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        # 🎨 Figure grande
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 📊 Normaliser les poids pour la taille
        sizes = (top_weights - top_weights.min()) / (top_weights.max() - top_weights.min()) * 400 + 100
        colors_scatter = plt.cm.Spectral(np.linspace(0, 1, len(top_words)))
        
        # 💫 Scatter plot avec bubble
        scatter = ax.scatter(range(len(top_words)), top_weights, s=sizes, 
                            alpha=0.6, c=colors_scatter, edgecolors='black', linewidth=2)
        
        # 📝 Texte centré sur les bulles
        for i, word in enumerate(top_words):
            ax.text(i, top_weights[i], word, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
        
        # 🔧 Configuration
        ax.set_xticks([])
        ax.set_ylabel('Weight', fontsize=14, fontweight='bold')
        ax.set_title(f'Topic #{topic_idx + 1} - Word Cloud (Top {num_words})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(f'output/topic_{topic_idx + 1}_word_cloud.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: topic_{topic_idx + 1}_word_cloud.png")
        plt.close()

# ============================================
# 🥧 FONCTION 4: Topic Distribution
# ============================================
def plot_topic_distribution_improved(lda, doc_term_matrix):
    """Camembert amélioré"""
    doc_topic_dist = lda.transform(doc_term_matrix)
    mean_dist = np.mean(doc_topic_dist, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 🎨 Couleurs attrayantes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7']
    
    # 🥧 Pie chart
    wedges, texts, autotexts = ax.pie(mean_dist, 
                                        labels=[f'Topic {i+1}' for i in range(len(mean_dist))],
                                        autopct='%1.1f%%',
                                        colors=colors[:len(mean_dist)],
                                        startangle=90,
                                        explode=[0.05]*len(mean_dist),
                                        textprops={'fontsize': 13, 'fontweight': 'bold'})
    
    # 📝 Amélioration des pourcentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title('Average Topic Distribution', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/topic_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: topic_distribution.png")
    plt.close()

# ============================================
# 🚀 MAIN - Exécution
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("📊 LDA Visualizations - IMPROVED VERSION")
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
    
    plot_top_words_by_topic(lda, vectorizer, num_words=12)
    plot_heatmap_improved(lda, doc_term_matrix)
    plot_word_clouds_improved(lda, vectorizer, num_words=15)
    plot_topic_distribution_improved(lda, doc_term_matrix)
    
    print("\n" + "="*60)
    print("🎉 All visualizations are ready!")
    print("="*60)
