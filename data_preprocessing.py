def load_data():
    corpus = [
        "We always come to Paris",
        "The professor is from Australia",
        "I live in Stanford",
        "He comes from Taiwan",
        "The capital of turkey is Ankara"
    ]
    return corpus

def preprocess_sentence(sentence):
    return sentence.lower().split()

def train_data():
    corpus = load_data()
    train_sentences = [preprocess_sentence(sent) for sent in corpus]
    return train_sentences

def train_labels():
    train_sentences = train_data()

    locations = set(["australia", "ankara", "paris", "stanford", "taiwan", "turkey"])
    
    train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]
    return train_labels