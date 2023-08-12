import tensorflow as tf
from Dataset import draw_random_sample,vocab_size
from BigramModel import BigramModel
from Parameters import block_size
import datetime
from Dataset import decode
import numpy as np


m = BigramModel(vocab_size)

idx, generation = decode(m.generate(tf.zeros((1,),tf.int64),20))
print(["".join(i) for i in generation.numpy()[:].astype(str)])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

logdir = "/Users/anu/PycharmProjects/TensorFlow2/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
epochs = 1
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step in range(10000):
        with tf.GradientTape() as tape:
            x,y = draw_random_sample(block_size)
            logits,loss = m(tf.reshape(x, (1, block_size - 1)), tf.reshape(y, (1, block_size - 1)))

        grads = tape.gradient(loss, m.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, m.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss at step %d: %.4f"
                % (step, float(loss))
            )
            print("Seen so far: %s samples" % ((step + 1)))

_, generation = decode(m.generate(tf.zeros((1,),tf.int64),300))

array = np.array(["".join(i) for i in generation.numpy()[:].astype(str)])
s = ''.join(array)
print(s)