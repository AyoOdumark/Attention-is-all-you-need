from typing import List, Dict
import pandas as pd

class CharacterTokenizer:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.vocabulary, self.vocab_map = self._create_vocabulary(self.corpus)
        self.vocabulary_length = len(self.vocabulary)
        
    def _create_token_map(self, tokens: set):
        tokens_map = {}
        for id, token in enumerate(tokens):
            tokens_map[token] = id
        
        return tokens_map
    
    def _create_vocabulary(self, corpus: List[str]) -> List[str]:
        # list_of_characters = [word.split() for sentence in corpus for word in sentence]
        characters = []
        for sentence in corpus:
            for character in sentence:
                characters.append(character)
        unique_tokens = set(characters)
        tokens_map = self._create_token_map(unique_tokens)
        
        return unique_tokens, tokens_map
    
    def encode(self, sentence: str) -> List[int]:
        tokens_ids = []
        list_of_characters = [word.split() for word in sentence]
        unique_tokens = set(list_of_characters)
        for token in unique_tokens:
            tokens_ids.append(self.vocab_map[token])
            
        return tokens_ids
    
    def decode(self, indexes: List[int]) -> List[str]:
        characters = []
        for id in indexes:
            characters.append(self.vocab_map.get(id))
        
        return characters
    