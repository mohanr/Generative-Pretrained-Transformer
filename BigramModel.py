import tensorflow as tf
from Block import Block
from keras.layers import Embedding
from Parameters import n_head,n_embd,block_size
import tensorflow_probability as tfp
from Dataset import vocab_size

class BigramModel(tf.keras.Model):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size,n_embd)
        self.position_embedding_table = Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(n_head, head_size) #Head(n_embd)
        self.blocks = Block(n_embd, n_head=n_head)

        self.ln_f = tf.keras.layers.LayerNormalization()  # final layer norm
        self.lm_head = tf.keras.layers.Dense(vocab_size, input_shape=(n_embd,), activation=None, use_bias=False)


    def call(self,idx,targets=None):
        # print(f'idx in call is {idx} and shape is {tf.shape(idx)}')
        B = 1
        if tf.size(tf.shape(idx)) == 1:
            T = tf.shape(idx)
        else:
            T = tf.shape(idx)[1]

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(tf.range(T))
        x = tf.add(tok_emb, tf.expand_dims(pos_emb,axis=0)) # (B,T,C)
        # print(f'Shape of tf.add(tok_emb, pos_emb) is {tf.shape(x)}')
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            bce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # loss = bce(targets,tf.squeeze(logits)).numpy()
            loss = bce(targets, tf.squeeze(logits))
        return logits, loss
    def generate(self,idx,max_new_tokens):
        i = tf.constant(0)
        c = lambda i, d: tf.less(i, max_new_tokens)

        def b(i, idx):
            # print(tf.shape(idx))
            idx_cond = idx[-block_size:]
            logits,loss = self(idx_cond)
            # print(f'Shape of logits is {tf.shape(logits)}')
            logits = logits[:,-1,:]
            probs = tf.nn.softmax(logits)
            # print(f'Shape of probs is {tf.shape(probs)}')
            idx_next = tfp.distributions.Multinomial(total_count=1,probs=probs)
            idx = tf.concat([idx,
                    tf.reshape(tf.squeeze(
                        tf.cast(tf.where(
                            tf.reshape(idx_next.sample(1),(vocab_size))),tf.int64))
                      ,(1,))],0)
            return tf.add(i, 1), idx

        _, idx = tf.while_loop(c, b, loop_vars=[i, idx])
        # print(f'idx in generate is {idx}')
        return idx
