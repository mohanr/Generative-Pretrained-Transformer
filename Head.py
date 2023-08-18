import numpy as np
import tensorflow as  tf

from Parameters import n_embd, dropout


class Head(tf.keras.Model):
    def __init__(self, head_size):
        super().__init__()
        self.key = tf.keras.layers.Dense(head_size, input_shape=(n_embd,), activation=None, use_bias=False)
        self.query = tf.keras.layers.Dense(head_size, input_shape=(n_embd,), activation=None, use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, input_shape=(n_embd,), activation=None, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        k = self.key(x)
        q = self.query(x)
        transpose = tf.transpose(k,perm=[0,2,1])
        matmul = tf.matmul(q,transpose)
        wei =  tf.divide(matmul, 1/tf.sqrt(tf.cast(n_embd,tf.float32)))
        tril = tf.linalg.band_part(wei, -1, 0)
        tril = tf.where(
                    tf.equal(tril,tf.constant(0, dtype=tf.float32)),
                    -np.inf,
                    tril)
        wei = tf.nn.softmax(tril,-1)
        # print(wei)

        v = self.value(x)
        out = tf.matmul(wei, v)
        # print(f'Shape of wei is {tf.shape(out)}')
        return out
