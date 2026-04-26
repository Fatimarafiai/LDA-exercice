# LDA Configuration
LDA_CONFIG = {
    'n_topics': 4,
    'max_iter': 20,
    'random_state': 42,
    'learning_method': 'online',
    'max_df': 0.95,
    'min_df': 2,
    'n_words': 10
}

# Paths
DATA_PATH = 'data/sample_documents.txt'
MODEL_PATH = 'models/lda_model.pkl'
RESULTS_PATH = 'results/'
