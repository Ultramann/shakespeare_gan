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
        '''
        Args:
            sentences (ShakespeareSentences)
            generator (Model):     Pre-compiled Keras model to generate character
                                   sequences from noise.
            discriminator (Model): Pre-compiled Keras model to distinguish between
                                   character sequences that are real and those
                                   faked by the generator.
            gen_dis (Model):       Keras model comprised of stacking the generator
                                   and discriminator for training end to end.
        '''
        self.sentences = sentences
        self.gen = generator
        self.dis = discriminator
        self.gen_dis = gen_dis

    def train(self, batch_size, nb_epochs, epoch_size='full'):
        for epoch in range(nb_epochs):
            sentences_batch_gen = self.sentences.get_batch_generator(batch_size,
                                                                     epoch_size)
            for batch in sentences_batch_gen:
                self._train_on_batch(batch)

    def _train_on_batch(self, batch):
        pass
