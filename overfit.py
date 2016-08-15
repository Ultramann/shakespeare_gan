import numpy as np
from training import GanTrainer
from data_processing import CharacterCoder
from modeling import make_models
import sys
import warnings
warnings.filterwarnings('ignore')
from keras import backend as K


class Sentences:
    def __init__(self, sentence='Fuck'):
        self.sentence = sentence
        self.chars = set(sentence)
        self.max_chars = len(sentence)

    def get_batch(self, batch_size):
        return np.array([self.sentence for _ in range(int(batch_size))])


class OverfitTrainer(GanTrainer):
    def __init__(self, sentences, char_coder, noise_dimension,
                       generator, discriminator, gan):
        super(OverfitTrainer, self).__init__(sentences, char_coder, noise_dimension,
                                             generator, discriminator, gan)


def set_up_trainer():
    sentences = Sentences()
    char_coder = CharacterCoder(sentences.chars)
    noise_dimension = 100
    models = make_models(noise_dimension,
                         sentences.max_chars,
                         len(sentences.chars))
    return OverfitTrainer(sentences, char_coder, noise_dimension, *models)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations


def check_gen_output(mt, noise):
    return mt.char_coder.decode(mt.gen.predict(noise)[0])


if __name__ == '__main__':
    steps = int(sys.argv[1])
    ot = set_up_trainer()
    ot.train(batch_size=1, nb_steps=steps, report_freq=10)
    test_noise = np.random.random((1,100))
