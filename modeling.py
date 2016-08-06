from keras.models import Sequential#, Activation
from keras.layers import LSTM, TimeDistributed, Dense

def make_models(char_dimension):
    generator = Sequential()
    generator.add(LSTM(64, input_dim=100, return_sequences=True))
    generator.add(TimeDistributed(Dense(char_dimension, activation='relu')))
    generator.compile(loss='softmax', optimizer='adam')

    discriminator = Sequential()
    discriminator.add(LSTM(64, input_dim=char_dimension))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')

    gen_dis = Sequential()
    gen_dis.add(generator)
    discriminator.trainable = False
    gen_dis.add(discriminator)
    gen_dis.compile(loss='binary_crossentropy', optimizer='adam')

    return generator, discriminator, gen_dis


class ModelTrainer:
    '''Class for handling the specific needs involving training a GAN.'''

    def __init__(self, sentences, generator, discriminator, gen_dis):
        self.sentences = sentences
        self.gen = generator
        self.dis = discriminator
        self.gen_dis = gen_dis

    def train(self, batch_size, epochs):
        pass
