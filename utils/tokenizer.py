from collections import Counter
import nltk

class LimitedTokenizer:
    def __init__(self, vocab_limit=1000):
        self.vocab_limit = vocab_limit
        self.word_to_index = {}
        self.index_to_word = {}
        self.oov_token = "[UNK]"
    
    def build_vocab(self, texts):
        """
        Builds a vocabulary based on the input texts with a limit on unique words.
        """
        # Tokenize and count word frequencies
        tokens = []
        for text in texts:
            tokens.extend(nltk.word_tokenize(text.lower()))
        
        word_counts = Counter(tokens)
        # Keep the most common words within the limit
        most_common = word_counts.most_common(self.vocab_limit)
        vocab = [word for word, _ in most_common]
        
        # Add words to vocabulary dictionary
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.word_to_index[self.oov_token] = len(vocab)  # Add OOV token
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
    
    def tokenize(self, text):
        """
        Tokenizes text into indices based on the limited vocabulary.
        """
        tokens = nltk.word_tokenize(text.lower())
        token_ids = [self.word_to_index.get(word, self.word_to_index[self.oov_token]) for word in tokens]
        return token_ids
    
    def pad_tokenize(self, text, length):
        token_ids = self.tokenize(text)
        return token_ids + [self.word_to_index[self.oov_token]] * (length - len(token_ids))
    
    def detokenize(self, token_ids):
        """
        Converts token indices back to words.
        """
        return [self.index_to_word.get(i, self.oov_token) for i in token_ids]
    
if __name__ == "__main__":
    # Example Usage
    texts = [
        "BERT is a powerful transformer model.",
        "This tokenizer limits the number of unique words in the vocabulary.",
        "Out-of-vocabulary words will be replaced with [UNK]."
    ]

    # Initialize and build tokenizer
    tokenizer = LimitedTokenizer(vocab_limit=10)
    tokenizer.build_vocab(texts)

    # Tokenize a sentence
    sentence = "BERT is amazing, but it can have out-of-vocabulary words!"
    token_ids = tokenizer.tokenize(sentence)
    detokenized = tokenizer.detokenize(token_ids)

    print("Vocabulary:", tokenizer.word_to_index)
    print("Token IDs:", token_ids)
    print("Detokenized:", detokenized)