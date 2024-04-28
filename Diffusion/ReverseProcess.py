import tensorflow as tf
from Forwardprocess import ForwardProcess
class ReverseProcess(ForwardProcess):
    def __init__(self, betas, model):
        super().__init__( betas )
        self.model = model
        self.T = len(betas) - 1

        self.sigma = tf.sqrt(
            tf.divide(
            tf.multiply((1 - self.alphas),
                        (1 - tf.roll(self.alpha_bar,1,axis=0))),
                (1 - self.alpha_bar)
        )
        )
        self.sigma = self.sigma.numpy()
        self.sigma[1] = 0.
        self.sigma = tf.convert_to_tensor(self.sigma)
        print(self.sigma)

    def get_x_t_minus_one(self, x_t, t):
        t_vector = tf.fill( tf.shape(x_t)[0:1],value=t)
        eps = self.model(x_t, t_vector)
        eps = tf.multiply( eps,tf.divide(( 1 - self.alphas[t]),
                                         tf.sqrt( 1 - self.alpha_bar[t])))
        mean = (tf.divide(1, (tf.multiply( tf.sqrt(self.alphas[t])
                                           ,tf.subtract(x_t, eps)))))
        # print(tf.multiply( tf.multiply( mean, self.sigma[t]),
        #                     tf.random.normal(tf.shape(x_t))))
        return tf.multiply( tf.multiply( mean, self.sigma[t]),
                            tf.random.normal(tf.shape(x_t)))
    # @tf.function
    def sample(self, n_samples = 1, full_trajectory = False ):
        x_t = tf.random.normal((n_samples,2))
        trajectory = [tf.identity(x_t)]
        for t in range(self.T, 1, -1):
            x_t = self.get_x_t_minus_one(x_t,t)
            if full_trajectory:
                trajectory.append(tf.identity(x_t))
        return tf.stack(trajectory, axis=0) if full_trajectory else x_t
