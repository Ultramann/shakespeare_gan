from keras.preprocessing.sequence import pad_sequences
import numpy as np


class GanTrainer:
    '''Class for handling the specific needs involving training a GAN.'''

    def __init__(self, sentences, char_coder, noise_dimension,
                       generator, discriminator, gan):
        '''
        Args:
            sentences (ShakespeareSentences)
            noise_dimension (int)
            generator (Model):     Pre-compiled Keras model to generate character
                                   sequences from noise
            discriminator (Model): Pre-compiled Keras model to distinguish between
                                   character sequences that are real and those
                                   artificially  by the generator
            gan (Model):           Keras model comprised of stacking the generator
                                   and discriminator for training end to end
        '''
        self.sentences = sentences
        self.char_coder = char_coder
        self.noise_dimension = noise_dimension
        self.gen = generator
        self.dis = discriminator
        self.gan = gan
        self.batch_size = 10
        self.fake = 1
        self.real = 0
        self._initialize_training()
        print('{: <7}  {: <10}  {: <10}  {: <11}  {: <11}'.format(
              'Step', 'Gen Loss', 'Dis Loss', 'Gen Changes', 'Dis Changes'))

    def _initialize_training(self):
        '''Helper method to get the GanTrainer ready for training.
        '''
        self.gen_loss = []
        self.gen_trained = []
        self.dis_loss = []
        self.dis_trained = []
        self._gen_step(update_weights=False)
        self._dis_step(update_weights=False)

    def train(self, batch_size, nb_steps, report_freq):
        '''
        Do training...and stuff? (I don't think there's much else to do...)

        Args:
            batch_size (int): Number of sentences GAN sees per step
            nb_steps (int): Number of training steps to train for
        '''
        evenize = lambda x: ((x + 1) // 2) * 2
        self.batch_size = evenize(batch_size)
        for step in range(nb_steps):
            self._train_batch()
            if step % report_freq == 0:
                self._report(report_freq)

    def _train_batch(self):
        '''
        Runs a single generator step and discriminator step of the GAN training
        process. Depending on the current generator loss and discriminator loss
        updates to the weights in those sub-models of the GAN will be updated.
        '''
        if self.gen_loss[-1] > 1:
            self._gen_step(update_weights=True)
            self._dis_step(update_weights=False)

        elif self.dis_loss[-1] > 1:
            self._dis_step(update_weights=True)
            self._gen_step(update_weights=False)

        else:
            self._dis_step(update_weights=True)
            self._gen_step(update_weights=True)

    def _gen_step(self, update_weights):
        '''
        Evaluate the generator on one batch of noise, track loss from batch.

        Args:
        update_weights (bool): If True, allow error from discriminator to update
                               the generator's weights. Aka, train the generator
        '''
        train_noise = np.random.random((self.batch_size, self.noise_dimension))
        train_target = np.full(self.batch_size, self.real, dtype=np.int8)

        if update_weights:
            current_loss = self.gan.train_on_batch(train_noise, train_target)
            was_trained = True
        else:
            current_loss = self.gan.test_on_batch(train_noise, train_target)
            was_trained = False

        self.gen_loss.append(current_loss)
        self.gen_trained.append(was_trained)

    def _dis_step(self, update_weights):
        '''
        Evaluate the discriminator on one batch of noise, track loss from batch.

        Args:
            update_weights (bool): If True, train the discriminator
        '''
        train_sentences, train_target = self._get_dis_training_sentences()

        if update_weights:
            self._set_dis_trainability(True)
            current_loss = self.dis.train_on_batch(train_sentences, train_target)
            self._set_dis_trainability(False)
            was_trained = True
        else:
            current_loss = self.dis.test_on_batch(train_sentences, train_target)
            was_trained = False

        self.dis_loss.append(current_loss)
        self.dis_trained.append(was_trained)

    def _get_dis_training_sentences(self):
        '''
        Helper method to produce real and fake character sequences for
        discriminator training.
        '''
        real_sentences = self.sentences.get_batch(self.batch_size/2)
        encoded_real_sentences = pad_sequences([self.char_coder.encode(sentence)
                                                for sentence in real_sentences],
                                               maxlen=self.sentences.max_chars,
                                               padding='post')
        generation_noise = np.random.random((self.batch_size/2,
                                             self.noise_dimension))
        encoded_fake_sentences = self.gen.predict_on_batch(generation_noise)

        labels = [np.full(self.batch_size/2, label, dtype=np.int8)
                  for label in [self.real, self.fake]]
        sentences = np.concatenate([encoded_real_sentences, encoded_fake_sentences])
        sentence_labels = np.concatenate(labels)

        return sentences, sentence_labels

    def _set_dis_trainability(self, trainable):
        '''Helper method to toggle trainability of discriminator.
        '''
        self.dis.trainable = trainable
        for layer in self.dis.layers:
            layer.trainable = trainable

    def _report(self, report_freq):
        '''
        Print 1 line containing most recent training status.
        Header for this table is printed in the object's constructor.
        '''
        n_steps = len(self.gen_loss)

        recent_gen_updates = sum(self.gen_trained[-report_freq:])
        recent_dis_updates = sum(self.dis_trained[-report_freq:])
        report_string = '{0:<7g}  {1:<10.4f}  {2:<10.4f}  {3:<11g}  {4:<11g}'
        print(report_string.format(n_steps, float(self.gen_loss[-1]),
                                   float(self.dis_loss[-1]), recent_gen_updates,
                                   recent_dis_updates))
