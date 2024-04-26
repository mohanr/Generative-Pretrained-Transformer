import tensorflow as tf

class ForwardProcess:

    def __init__(self, betas):
        super().__init__()
        self.betas = betas
        self.alphas = 1. - betas
        self.alpha_bar = tf.math.cumprod(self.alphas,axis=-1)

    def get_x_t(self, x_0, t):
        eps_0 = tf.random.normal(tf.shape(x_0))
        if tf.shape(t).numpy().size == 0 :
            # A different shape is needed for plotting
            alpha_bar = self.alpha_bar[t, None]
        else:
            alpha_bar = tf.gather(self.alpha_bar,tf.cast(t,dtype=tf.int32))[:, tf.newaxis]
        mean = tf.multiply( tf.cast(tf.sqrt( alpha_bar ),dtype=tf.float32),
                            tf.cast(x_0,dtype=tf.float32))
        std = tf.sqrt( 1 - alpha_bar)
        return (eps_0, tf.add( mean, tf.multiply( std, eps_0)))