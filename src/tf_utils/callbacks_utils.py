import tensorflow as tf


class SWA(tf.keras.callbacks.Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        if self.nb_epoch <= self.swa_epoch:
            raise ValueError('Training ends before stochastic weight averaging begins, swa_epoch ({}) has to be '
                             'smaller than training epochs ({})'.format(self.swa_epoch, self.nb_epoch))

        print('Stochastic weight averaging selected for last {} epochs.'.format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
        elif epoch > self.swa_epoch:
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i]) \
                                      / ((epoch - self.swa_epoch) + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
