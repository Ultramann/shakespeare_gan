from data_processing import ShakespeareSentences, CharacterCoder
from modeling import make_models
from training import GanTrainer
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from keras import backend as K


def set_up_trainer():
    sentences = ShakespeareSentences()
    char_coder = CharacterCoder(sentences.chars)
    noise_dimension = 300
    models = make_models(noise_dimension,
                         sentences.max_chars,
                         len(sentences.chars))
    model_trainer = GanTrainer(sentences, char_coder, noise_dimension, *models)
    return model_trainer


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations


def check_gen_output(mt, noise):
    return mt.char_coder.decode(mt.gen.predict(noise)[0])


if __name__ == '__main__':
    batch_size, steps = int(sys.argv[1]), int(sys.argv[2])
    mt = set_up_trainer()
    mt.train(batch_size=batch_size, nb_steps=steps, report_freq=10)
    test_noise = np.random.random((1,300))
