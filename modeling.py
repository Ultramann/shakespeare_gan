from keras.layers import LSTM, TimeDistributed, Dense, Input, RepeatVector, Layer
from keras.models import Model
from keras import backend as K


def make_models(noise_dimension, max_chars, char_dimension):
    '''
    Args:
        noise_dimension (int)
        max_chars (int)
        char_dimension (int)

    Returns:
        generator (Model):     Takes noise vector and returns a max_chars long
                               sequence of char_dimensioned vectors
        discriminator (Model): Takes a sequence and classifies it as real or
                               artificially generated text
        GAN (Model):           Staked generator and discriminator
    '''
    gen_input = Input(shape=(noise_dimension,), name='noise')
    gen = RepeatVector(max_chars, name='lots_of_noise')(gen_input)
    gen = LSTM(64, return_sequences=True, name='gen_lstm_1')(gen)
    gen = LSTM(64, return_sequences=True, name='gen_lstm_2')(gen)
    gen = TimeDistributed(Dense(char_dimension, activation='softmax'),
                          name='gen_out')(gen)
    gen_output = TimeDistributed(OneHot(), name='one_hot')(gen)
    generator = Model(input=gen_input, output=gen_output, name='generator')
    generator.compile(loss='mae', optimizer='adam')
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

    gan_input = Input(shape=(noise_dimension,), name='gan_noise')
    gan_gen = generator(gan_input)
    gan_dis = discriminator(gan_gen)
    gan = Model(input=gan_input, output=gan_dis)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    print('\t\t GAN Summary:')
    gan.summary()

    return generator, discriminator, gan


def one_hot(x):
    '''
    Sparse-ifies 3-dimensional tensor by making the largest value 1 and the rest 0.
    Aka, make one hot.

    Args: 
        x (3d theano tensor)

    Returns:
       3d theano tensor 
    '''
    return K.cast(K.equal(K.max(x, axis=-1, keepdims=True), x), K.floatx())


class OneHot(Layer):
    def __init__(self, **kwargs):
        self.uses_learning_phase = True
        super(OneHot, self).__init__(**kwargs)

    def call(self, x, mask=None):
        x = K.in_test_phase(one_hot(x), x)
        #x = K.in_test_phase(one_hot(x), one_hot(x))
        return x

    def get_config(self):
        return super(OneHot, self).get_config()
