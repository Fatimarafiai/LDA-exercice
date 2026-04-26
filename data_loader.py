
import os

def load_documents(file_path='data/sample_documents.txt'):
    """Charge les documents depuis un fichier texte"""
    if not os.path.exists(file_path):
        print(f"Erreur: {file_path} n'existe pas!")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    print(f"✅ {len(documents)} documents chargés")
    return documents

def preprocess_documents(documents):
    """Preprocessing basique des documents"""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Télécharger les ressources NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    processed = []
    
    for doc in documents:
        tokens = word_tokenize(doc.lower())
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
        processed.append(' '.join(filtered))
    
    return processed
