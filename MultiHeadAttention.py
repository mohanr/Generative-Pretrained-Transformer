import tensorflow as tf
from Head import Head
from Parameters import n_embd,dropout

class MultiHeadAttention(tf.keras.Model):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = tf.keras.layers.Dense(n_embd, input_shape=(n_embd,), activation=None, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads],-1)
        out = self.dropout(self.proj(out))
        return out
