import tensorflow as tf

class ForwardProcess(tf.keras.Model):

    def __init__(self, betas):
        super().__init__()
        self.betas = betas
        self.alphas = 1. - betas
        self.alpha_bar = tf.math.cumprod(self.alphas,axis=-1)

    def call(self, x_0, t):
        eps_0 = tf.random.normal(x_0)
        alpha_bar = self.alpha_bar[t, None]
        mean = tf.multiply( tf.sqrt( alpha_bar ), x_0)
        std = tf.sqrt( 1 - alpha_bar)
        return (eps_0, tf.add( mean, tf.multiply( std, eps_0)))