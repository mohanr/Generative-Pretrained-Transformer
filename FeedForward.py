import tensorflow as tf
from Parameters import dropout

class FeedForward(tf.keras.Model):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(4 * n_embd, input_shape=(None,n_embd), activation=None, use_bias=False),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(n_embd, input_shape=(4 * n_embd,), activation=None, use_bias=False),
                tf.keras.layers.Dropout(dropout)
            ]
        )

    def call(self, x):
        return self.net(x)
