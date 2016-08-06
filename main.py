from data_processing import load_shakespeare_sentences
from modeling import make_models, ModelTrainer

def run_train():
    sentences = load_shakespeare_sentences()
    models = make_models()
    model_trainer = ModelTrainer(sentences, *models)
    model_trainer.train(epochs=10, batch_size = 20)

if __name__ == '__main__':
    run_train()
