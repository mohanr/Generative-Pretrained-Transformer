import tensorflow as tf
class NoisePredictor(tf.keras.Model):

    def __init__(self, T):
        super().__init__()
        self.T = T
        self.t_encoder = tf.keras.layers.Dense(1,input_shape=(T,),activation=None)
        self.net = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense( 100, input_shape=(3,), activation=None),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(100, activation=None),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(100,  activation=None),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(20, activation=None),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense( 2, activation=None)]

        )

    def call(self, x_t, t):
        t_embedding = self.t_encoder(tf.one_hot( t - 1, self.T))
        inp = tf.concat([x_t, t_embedding], axis=1)
        return self.model(inp)