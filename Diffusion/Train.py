import tensorflow as tf
from Forwardprocess import ForwardProcess
from NoisePredictor import NoisePredictor
from ReverseProcess import ReverseProcess
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
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2,
                                     beta_1=.9,
                                     beta_2=.999,
                                     decay=1e-4)
N = tf.shape(X)[0]

epochs = 10000
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
rp = ReverseProcess(betas,model)
samples = rp.sample(1000)
# samples = tf.reshape(rp.sample(1000),(1000,2))
print(tf.shape(samples))
fig,ax = plt.subplots(figsize=(5,5))
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),aspect="equal")
sns.kdeplot(x = samples[:,0],y = samples[:,1],ax = ax,fill=True,thresh=0.,cut=15.,clip=(-3,3),bw_adjust=0.5)
sns.scatterplot(x = samples[:100][:,0],y = samples[:100][:,1],ax = ax)
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),aspect="equal")
plt.show()