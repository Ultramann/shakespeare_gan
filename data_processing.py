import os
import re
import numpy as np
from numpy.random import random_integers


def load_file(file_path):
    '''Helper function for loading contents at file path.'''

    with open(file_path) as f:
        return f.read()


class ShakespeareSentences:
    '''
    Class to simplify getting random batches of Shakespeare sentences.
    '''
    def __init__(self, text_folder='clean_text/', end_token='#'):
        self.end_token = end_token
        self.chars = set(end_token)
        self.max_chars = 0
        self.sentences = self._get_sentences(text_folder)
        self.total_sentences = len(self.sentences)

    def __repr__(self):
        sentence_message = 'There are a total of {} Shakespeare sentences.'
        return sentence_message.format(self.total_sentences)

    def _get_sentences(self, text_folder):

        def sentenciate(text_path):
            '''
            Helper function for making sentences from text in
            file at text_path.
            '''
            text = load_file(os.path.join(text_folder, text_path))
            sentences_n_punct = re.split(r'(\.|\?|\!)', text)
            full_sentences = []
            for i in range(0, len(sentences_n_punct)-1, 2):
                sentence = sentences_n_punct[i] + sentences_n_punct[i+1]
                self.chars.update(set(sentence))
                self.max_chars = max(len(sentence), self.max_chars)
                full_sentences.append(sentence.rstrip().lstrip() + self.end_token)
            return full_sentences

        text_files = os.listdir(text_folder)
        sentences = np.array([sentence for text_path in text_files
                                       for sentence in sentenciate(text_path)])
        return sentences

    def get_batch(self, batch_size):
        '''
        Args:
            batch_size (int): Number of random sentences to get
        '''
        random_idxs = random_integers(0, self.total_sentences, batch_size)
        random_sentences = self.sentences[random_idxs]
        return random_sentences


class CharacterCoder:
    '''Class for (en/de)coding characters to/from one-hot vector representation.'''

    def __init__(self, chars):
        '''
        Args:
            chars (set): All of the characters that the CharacterCoder will
                         be able to (en/de)code
        '''
        self.chars = sorted(chars)
        self.char_hash = {c: i for i, c in enumerate(self.chars)}
        self.idx_hash = {i: c for i, c in enumerate(self.chars)}
        self.vector_len = len(self.chars)

    def encode(self, chars):
        '''
        Args:
            chars (str): Characters to encoded to one-hot vectors

        Returns:
            ndarray: 2D array with one-hot vectors corresponding to the character
                     at each position
        '''
        char_vectors = np.zeros((len(chars), self.vector_len))
        for i, char in enumerate(chars):
            char_vectors[i, self.char_hash[char]] = 1
        return char_vectors

    def decode(self, char_vectors):
        '''
        Args:
            char_vectors (ndarray): 2D array of corresponding to the characters

        Returns:
            str: Decoded string

        '''
        char_idxs = char_vectors.argmax(axis=-1)
        return ''.join(self.idx_hash[idx] for idx in char_idxs)
