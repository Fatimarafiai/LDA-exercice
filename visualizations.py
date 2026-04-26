import matplotlib.pyplot as plt
from visualizations import plot_topics, plot_word_clouds, plot_heatmap
from main import *

# Charger et traiter les données
documents = load_documents()
processed_docs = preprocess_documents(documents)
vectorizer, lda, doc_term_matrix = train_lda_model(processed_docs)

# Créer les visualisations
print("\n📊 Création des visualisations...")

# 1. Top words par topic
plot_topics(lda, vectorizer, num_words=10)
plt.tight_layout()
plt.savefig('output/top_words_by_topic.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: top_words_by_topic.png")

# 2. Heatmap
plot_heatmap(lda, doc_term_matrix)
plt.tight_layout()
plt.savefig('output/document_topic_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: document_topic_heatmap.png")

print("\n🎉 Toutes les visualisations sont prêtes !")
plt.show()
