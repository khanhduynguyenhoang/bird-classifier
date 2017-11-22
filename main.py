import tensorflow as tf
import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

x_train = np.zeros([0, 1024 * 3])
y_train = np.zeros([0])
for batch_number in range(1, 6):
    data = unpickle("data/cifar-10-batches-py/data_batch_" + str(batch_number))
    x = np.asarray(data[b"data"])
    y = np.asarray(data[b"labels"])
    x_train = np.append(x_train, x, axis = 0)
    y_train = np.append(y_train, y, axis = 0)
print(x_train.shape)
print(y_train.shape)

test_data = unpickle("data/cifar-10-batches-py/test_batch")
x_test = np.asarray(test_data[b"data"])
y_test = np.asarray(test_data[b"labels"])
print(x_test.shape)
print(y_test.shape)

r = x_train[:, : 1024]
g = x_train[:, 1024: 2048]
b = x_train[:, 1024 * 2: 1024 * 3]

x_train = (r * 0.299 + g * 0.587 + b * 0.114).astype(int)
x_test = (x_test[:, : 1024] * 0.299 + x_test[:, 1024: 1024 * 2] * 0.587 + x_test[:, 1024 * 2: 1024 * 3] * 0.114).astype(int)

print(x_train.shape)
print(x_test.shape)

#y_train[x == 2] = -1
# y_test[x == 2] = -1
# y_train[x >= 0] = 0
# y_test[x >= 0] = 0
# y_train[x < 0] = 1
# y_test[x < 0] = 1

tf.reset_default_graph()

IMG_SIZE = 64

# Training Parameters
learning_rate = 0.001
num_steps = 1500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 1024
num_classes = 10
dropout_rate=0.3

# tf Graph input
x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.int32, [None])
training = tf.placeholder_with_default(False, shape=(), name='training')


def conv_net(x):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

    conv1 = tf.layers.conv2d(x, filters=256, kernel_size=3, padding="SAME",
                             activation=tf.nn.relu, name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    flat_layer = tf.reshape(pool1, shape=[-1, 256 * 16 * 16])

    fc1 = tf.layers.dense(flat_layer, 1024, activation=tf.nn.relu, name="fc1")

    fc1_drop = tf.layers.dropout(fc1, dropout_rate, training=training)

    out = tf.layers.dense(fc1_drop, num_classes, name="output")

    return out

def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch

pred = conv_net(x)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer()
training_op=optimizer.minimize(cost)

correct = tf.nn.in_top_k(pred, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()


sess= tf.Session()

sess.run(init)
for step in range(1, num_steps+1):
    batch_x, batch_y = random_batch(x_train, y_train, batch_size)
    sess.run(training_op, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0 or step == 1:
        acc = sess.run( accuracy, feed_dict={x: batch_x,y: batch_y})
        print('Step:',step, ', Accuracy:',acc)
        print('Cost is: ', sess.run( cost, feed_dict={x: batch_x,y: batch_y}))


print("Optimization Finished!")

batch_test = 100
repeat = int(len(x_test) / batch_test)
cur = 0
for index in range(repeat):
    test_acc = sess.run(accuracy, feed_dict={x: x_test[index * batch_test: (index + 1) * batch_test],
                                             y: y_test[index * batch_test: (index + 1) * batch_test]})
    cur = cur + test_acc

cur = cur / repeat

print("Testing Accuracy:", cur)

