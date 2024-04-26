import tensorflow as tf
from Forwardprocess import ForwardProcess
from NoisePredictor import NoisePredictor
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns

X,y = make_moons(n_samples=1000)

print(tf.shape(y))

def linear_scale( X )  :
    X = tf.convert_to_tensor(X)
    X -= tf.reduce_min(X,axis=0,keepdims=True)
    X /= tf.reduce_max(X,axis=0,keepdims=True)
    X = X * 2 - 1

print(tf.shape(X))

T = 10
betas = tf.pow( tf.linspace( 0., .99, T + 1), 4 )
fp = ForwardProcess( betas )

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for idx, t in enumerate([0, 2, T]):
    x_t = fp.get_x_t( X[:100], tf.convert_to_tensor(t))[1].numpy()
    sns.scatterplot(x = x_t[:, 0],y = x_t[:, 1], ax = ax[idx])
    ax[idx].set(xlim=(-1.5, 1.5),ylim=(-1.5, 1.5))
plt.show()

model = NoisePredictor( 10 )
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
N = tf.shape(X)[0]

epochs = 1 #5000
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    t = tf.random.uniform(minval=1, maxval=T + 1, shape=(N,))
    eps_0, x_t = fp.get_x_t(tf.convert_to_tensor(X), t)
    for step in range(1):
        with tf.GradientTape() as tape:
            pred_eps = model(x_t,t)
            loss = tf.reduce_mean(tf.square( pred_eps - eps_0))
            grads = tape.gradient(loss, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 20 == 0:
                print(
                    "Training loss at step %d: %.4f"
                    % (step, float(loss))
                )
