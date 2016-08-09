from data_processing import ShakespeareSentences, CharacterCoder
from modeling import make_models, GanTrainer

def run_train():
    sentences = ShakespeareSentences()
    char_coder = CharacterCoder(sentences.chars)
    noise_dimension = 100
    models = make_models(noise_dimension,
                         len(sentences.chars),
                         sentences.max_chars)
    model_trainer = GanTrainer(sentences, char_coder, noise_dimension, *models)
    model_trainer.train(batch_size=20, nb_steps=100)
    return model_trainer

if __name__ == '__main__':
    mt = run_train()
