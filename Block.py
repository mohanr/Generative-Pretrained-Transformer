import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class Block(tf.keras.Model):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = tf.add(x , self.sa(self.ln1(x)))
        x = tf.add(x , self.ffwd(self.ln2(x)))
        return x
