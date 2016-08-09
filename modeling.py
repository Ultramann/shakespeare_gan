from keras.layers import LSTM, TimeDistributed, Dense, Input
from keras.models import Model
import numpy as np

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

        self.gen_loss = []
        self.gen_trained = []
        self.dis_loss
        self.dis_trained = []

    def train(self, batch_size, nb_steps):
        evenize = lambda x: ((x + 1) // 2) * 2
        self.batch_size = evenize(batch_size)
        for step in range(nb_steps):
            self._train_on_batch()

    def _train_on_batch(self):
        if self.gen_loss[-1] > 1:
            self._train_gen(update_weights=True)
            self._train_dis(update_weights=False)
        elif self.dis_loss[-1] > 1:
            self._train_dis(update_weights=True)
            self._train_gen(update_weights=False)
        else:
            self._train_dis(update_weights=True)
            self._train_gen(update_weights=True)

    def _train_gen(self, update_weights):

        train_noise = np.random.random((self.batch_size, self.noise_dimension))
        train_target = np.ones(self.batch_size)

        if update_weights:
            current_loss = self.gen.train_on_batch(train_noise, train_target)
            was_trained = True
        else:
            current_loss = self.gen.test_on_batch(train_noise, train_target)
            was_trained = False

        self.gen_loss.append(current_loss)
        self.gen_trained.append(was_trained)

    def _train_dis(self, update_weights):

        train_text, train_target = self._get_dis_training_text()

        if update_weights:
            self._set_dis_trainability(True)
            current_loss = self.dis.train_on_batch(train_text, train_target)
            self._set_dis_trainability(False)
            was_trained = True
        else:
            current_loss = self.dis.test_on_batch(train_text, train_target)
            was_trained = False

        self.dis_loss.append(current_loss)
        self.dis_trained.append(was_trained)

    def _get_dis_training_text(self):
        pass

    def _set_dis_trainability(self, trainable):
        self.dis.trainable = trainable
        for layer in self.dis.layers:
            layer.trainable = trainable
