import tensorflow as tf
import numpy as np
from NoisePredictor import NoisePredictor
from sklearn.datasets import make_moons

model = NoisePredictor( 10 )
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
X,y = make_moons(n_samples=1000)

def linear_scale( X )  :
    X = tf.convert_to_tensor(X)
    X -= tf.reduce_min(X,axis=0,keepdims=True)
    X /= tf.reduce_max(X,axis=0,keepdims=True)
    X = X * 2 - 1

# print(tf.shape(X))
# print(X)
print(linear_scale([1.,2.,3.,4.,5.]))