from keras.callbacks import Callback

class CocoEval(Callback):

    def __init__(self, neg1, gen2):
        pass

    def on_batch_begin(self, batch, logs=None):
        #print("\n\ncallback: on_batch_begin", batch, logs)
        pass

    def on_batch_end(self, batch, logs=None):
        #print("\n\ncallback: on_batch_end", batch, logs)
        pass

    def on_epoch_end(self, epoch, logs=None):
        #print("\n\ncallback: on_epoch_end", epoch, logs)
        pass

