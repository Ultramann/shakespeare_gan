from keras.layers import LSTM, TimeDistributed, Dense, Input
from keras.models import Model
from numpy import random

def make_models(noise_dimension, char_dimension, max_chars):
    '''
    Args:
        noise_dimension (int)
        char_dimension (int)
        max_chars (int)

    Returns:
        generator (Model):     Takes noise vector and returns a max_chars long
                               sequence of char_dimensioned vectors
        discriminator (Model): Takes a sequence and classifies it as real or
                               artificially generated text
        GAN (Model):           Staked generator and discriminator
    '''
    gen_input = Input(shape=(max_chars, noise_dimension), name='noise')
    gen = LSTM(64, return_sequences=True, name='gen_lstm_1')(gen_input)
    gen = LSTM(64, return_sequences=True, name='gen_lstm_2')(gen)
    gen_output = TimeDistributed(Dense(char_dimension, activation='softmax'),
                                 name='gen_out')(gen)
    generator = Model(input=gen_input, output=gen_output, name='generator')
    generator.compile(loss='categorical_crossentropy', optimizer='adam')
    print('\t\t Generator Summary:')
    generator.summary()

    seq_input = Input(shape=(max_chars, char_dimension), name='text')
    dis = LSTM(64, return_sequences=True, name='dis_lstm_1')(seq_input)
    dis = LSTM(64, name='dis_lstm_to_vec')(dis)
    dis_output = Dense(1, activation='sigmoid', name='dis_out')(dis)
    discriminator = Model(input=seq_input, output=dis_output, name='discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    print('\t\t Discriminator Summary:')
    discriminator.summary()

    gan_input = Input(shape=(max_chars, noise_dimension), name='gan_noise')
    gan_gen = generator(gan_input)
    gan_dis = discriminator(gan_gen)
    gan = Model(input=gan_input, output=gan_dis)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    print('\t\t GAN Summary:')
    gan.summary()

    return generator, discriminator, gan


class GanTrainer:
    '''Class for handling the specific needs involving training a GAN.'''

    def __init__(self, sentences, noise_dimension, generator, discriminator, gan):
        '''
        Args:
            sentences (ShakespeareSentences)
            noise_dimension (int)
            generator (Model):     Pre-compiled Keras model to generate character
                                   sequences from noise
            discriminator (Model): Pre-compiled Keras model to distinguish between
                                   character sequences that are real and those
                                   faked by the generator
            gan (Model):           Keras model comprised of stacking the generator
                                   and discriminator for training end to end
        '''
        self.sentences = sentences
        self.noise_dimension = noise_dimension
        self.gen = generator
        self.dis = discriminator
        self.gan = gan
        self.batch_size = None

    def train(self, batch_size, nb_steps):
        self.batch_size = batch_size
        for step in range(nb_steps):
            self._train_on_batch()

    def _train_on_batch(self, sentences):
        if self.gen_loss > 1:
            self._train_gen(True)
            self._train_dis(False)
        elif self.dis_loss > 1:
            self._train_dis(True)
            self._train_gen(False)
        else:
            self._train_dis(True)
            self._train_gen(True)

    def _train_gen(self, update_weights):
       train_noise = random.random((self.batch_size, self.noise_dimension))

    def _train_dis(update_weights):
        pass

    def set_dis_trainability(self, ability):
        self.dis.trainable = ability
        for layer in self.dis.layers:
            layer.trainable = ability
