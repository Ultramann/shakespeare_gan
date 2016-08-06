import os
import re
import numpy as np


def load_file(file_path):
    '''Helper function for loading contents at file path.'''

    with open(file_path) as f:
        return f.read()


class ShakespeareSentences:
    '''
    Class to simplify getting non-repeating, random sets of Shakespeare sentences.
    '''
    def __init__(self, text_folder='clean_text/',
                 start_token='%', end_token='#'):

        self.start_token = start_token
        self.end_token = end_token
        self.chars = set((start_token, end_token))
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
                full_sentences.append(self.start_token + 
                                      sentence.strip() +
                                      self.end_token)
            return full_sentences

        text_files = os.listdir(text_folder)
        sentences = np.array([sentence for text_path in text_files 
                                       for sentence in sentenciate(text_path)])
        return sentences

    def get_batch_generator(self, batch_size, epoch_size):
        '''
        Method to get generator that yields batches of non-repeating
        Shakespeare sentences.

        Args:
            batch_size (int):     Number of sentences per batch
            epoch_size (int/str): How many sentences to go through in an epoch
        '''

        def sentence_batch_gen(rand_sentences):
            '''Generator that yields batches of random sentences.'''
            for i in range(0, len(rand_sentences), batch_size):
                sentence_batch = rand_sentences[i:i+batch_size]
                yield sentence_batch

        if epoch_size == 'full':
            epoch_size = self.total_sentences

        idxs = np.arange(self.total_sentences)
        np.random.shuffle(idxs)
        random_idxs = idxs[:epoch_size]
        random_sentences = self.sentences[random_idxs]
        return sentence_batch_gen(random_sentences)

        
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
