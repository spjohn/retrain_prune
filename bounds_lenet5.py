'''
Author: Soumya Sara John
Requires: tensorflow==1.15
'''

from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
'''
Input the data
'''
start_time = time.time()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
Define parameters
'''
learning_rate = 0.01
training_iters = 1000
batch_size = 128
display_step = 100

n_input = 784
n_classes = 10
dropout = 0.75

'''
Define the variables
'''
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16])),
    'wd1': tf.Variable(tf.random_normal([7*7*16, 84])),
    'out': tf.Variable(tf.random_normal([84, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([6])),
    'bc2': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([84])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

'''
Define the layers of the network
'''
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def sparse_conv(weights,p):
    total_no = weights.size
    sparse_no = int(total_no*p)
    for i in range(sparse_no):
        pos0 = random.randint(0,weights.shape[0]-1)
        pos1 = random.randint(0,weights.shape[1]-1)
        pos2 = random.randint(0,weights.shape[2]-1)
        pos3 = random.randint(0,weights.shape[3]-1)
        weights[pos0][pos1][pos2][pos3] = 0
    return weights

def sparse_fc(weights,p):
    total_no = weights.size
    sparse_no = int(total_no*p)
    for i in range(sparse_no):
        pos0 = random.randint(0,weights.shape[0]-1)
        pos1 = random.randint(0,weights.shape[1]-1)
        weights[pos0][pos1] = 0
    return weights

x1 = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = conv2d(x1, weights['wc1'], biases['bc1'])
conv1 = maxpool2d(conv1, k=2)
conv1 = tf.nn.relu(conv1)

conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

conv2 = maxpool2d(conv2, k=2)
conv2 = tf.nn.relu(conv2)

fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc2 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc = tf.nn.relu(fc2)

out = tf.add(tf.matmul(fc, weights['out']), biases['out'])

'''
Define train operations
'''
pred = out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads = optimizer.compute_gradients(cost, var_list = weights)
grads_ = optimizer.compute_gradients(cost)
train_op = optimizer.apply_gradients(grads_)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
Prune in steps and retrain. Store the required values.
'''
init = tf.global_variables_initializer()
mu = 0.0001
epsilon=1
t_init=training_iters
with tf.Session() as sess:
  sess.run(init)
  g_init = sess.run(grads, feed_dict = {x: mnist.test.images[:32],y: mnist.test.labels[:32],keep_prob: 1.})
  batch_x, batch_y = mnist.train.next_batch(batch_size)
  for step in range(1,training_iters+1):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      sess.run(train_op, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
      if step % display_step == 0:
          loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
          print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}%".format(acc*100))
  print("Optimization Finished!")

  test_loss_original, acc_final = sess.run([cost,accuracy], feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.})
  g_final = sess.run(grads, feed_dict = {x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.})
  print("Testing Accuracy:",acc_final)
  final_weights_c1 = weights['wc1'].eval(sess)
  final_weights_c2 = weights['wc2'].eval(sess)
  final_weights_d1 = weights['wd1'].eval(sess)
  final_weights_out = weights['out'].eval(sess)
  final_biases_c1 = biases['bc1'].eval(sess)
  final_biases_c2 = biases['bc2'].eval(sess)
  final_biases_d1 = biases['bd1'].eval(sess)
  final_biases_out = biases['out'].eval(sess)

  pr=10
  itr=50
  num_iter_list = np.zeros((pr,itr))
  lower_hard_bound=np.zeros((pr,itr))
  upper_hard_bound=np.zeros((pr,itr))
  test_acc_list=np.zeros((pr,itr))
  num_nonzero_before_retraining = np.zeros((pr,itr))
  num_nonzero_after_retraining = np.zeros((pr,itr))
  gamma_series = np.zeros((pr,itr))
  grad_init = 0
  for gi in g_init:
    grad_init = grad_init + np.square(np.linalg.norm(gi[0]))

  for p in range(1,pr):
    pop = p/10
    for i in range(itr):
      print('p: ',p, 'i: ', i)
      weights_sparse_c1 = sparse_conv(np.array(final_weights_c1),p=pop)
      weights_sparse_c2 = sparse_conv(np.array(final_weights_c2),p=pop)
      weights_sparse_d1 = sparse_fc(np.array(final_weights_d1),p=pop)
      weights_sparse_out = sparse_fc(np.array(final_weights_out),p=pop)
      num_nonzero_before_retrain = np.count_nonzero(weights_sparse_c1) + np.count_nonzero(weights_sparse_c2) + np.count_nonzero(weights_sparse_d1) + np.count_nonzero(weights_sparse_out)
      num_nonzero_before_retraining[p][i] = num_nonzero_before_retrain
      sess.run(init)
      sess.run(tf.assign(weights['wc1'],weights_sparse_c1))
      sess.run(tf.assign(weights['wc2'],weights_sparse_c2))
      sess.run(tf.assign(weights['wd1'],weights_sparse_d1))
      sess.run(tf.assign(weights['out'],weights_sparse_out))
      sess.run(tf.assign(biases['bc1'], final_biases_c1))
      sess.run(tf.assign(biases['bc2'], final_biases_c2))
      sess.run(tf.assign(biases['bd1'], final_biases_d1))
      sess.run(tf.assign(biases['out'], final_biases_out))
      g_sparse = sess.run(grads, feed_dict = {x: mnist.test.images[:32],y: mnist.test.labels[:32],keep_prob: 1.})
      test_loss, acc_sparse = sess.run([cost,accuracy], feed_dict = {x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.})
      delta_w = np.square(np.linalg.norm(final_weights_c1 - weights_sparse_c1))+\
                np.square(np.linalg.norm(final_weights_c2 - weights_sparse_c2))+\
                np.square(np.linalg.norm(final_weights_d1 - weights_sparse_d1))+\
                np.square(np.linalg.norm(final_weights_out - weights_sparse_out))
      grad_sparse = 0
      for gs in g_sparse:
          grad_sparse = grad_sparse + np.square(np.linalg.norm(gs[0]))
      gamma_series[p][i] = delta_w*grad_sparse/(2*epsilon*learning_rate*t_init*grad_init)
      lower_hard_bound[p][i] = grad_sparse/grad_init
      upper_hard_bound[p][i] = delta_w/(2*epsilon*learning_rate)
      print('Retraining...')
      num_iter = 0
      while(np.abs(test_loss - test_loss_original) > epsilon and num_iter<150):
        num_iter = num_iter + 1
        # print(num_iter)
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x:batch_x,y:batch_y})
        loss1,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob: dropout})
        test_loss = sess.run(cost,feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.})
      weights_c1 = weights['wc1'].eval(sess)
      weights_c2 = weights['wc2'].eval(sess)
      weights_d1 = weights['wd1'].eval(sess)
      weights_out = weights['out'].eval(sess)
      num_iter_list[p][i] = num_iter
      num_nonzero_after_retrain = np.count_nonzero(weights_c1) + np.count_nonzero(weights_c2) + np.count_nonzero(weights_d1) + np.count_nonzero(weights_out)
      num_nonzero_after_retraining[p][i] = num_nonzero_after_retrain
      test_loss, test_acc = sess.run([cost,accuracy], feed_dict = {x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.})
      print('num_iter: ', num_iter)
      test_acc_list[p][i] = test_acc

'''
Calculate the bounds
'''
f1 = np.zeros((pr,1))
f2 = np.zeros((pr,1))
for i in range(1,pr):
  f1[i] = 1e-4*i*np.abs(np.log(i/10))
  f2[i] = 1e-2
upper_bound = np.multiply(f1, upper_hard_bound)
gammas = np.multiply(f2, gamma_series)
lower_bound = np.multiply(gammas, lower_hard_bound)

'''
Plot and save the values
'''
fig, ax = plt.subplots(1)
t = [10,20,30,40,50,60,70,80,90]
num_iter_retrain_mu = np.mean(num_iter_list[1:], axis=1)
num_iter_retrain_std = np.std(num_iter_list[1:], axis=1)
ax.plot(t,np.flip(num_iter_retrain_mu), label='Experimentally obtained', color='red')
# ax.fill_between(t, num_iter_retrain_mu+num_iter_retrain_std, num_iter_retrain_mu-num_iter_retrain_std, facecolor='blue', alpha=0.2)

lower_hard_bound_mu = np.mean(lower_bound[1:], axis=1)
lower_hard_bound_std = np.std(lower_bound[1:], axis=1)
ax.plot(t,lower_hard_bound_mu, label='Calculated lower bound', color='blue', linestyle='--')
# ax.fill_between(t, lower_hard_bound_mu+lower_hard_bound_std, lower_hard_bound_mu-lower_hard_bound_std, facecolor='orange', alpha=0.2)

upper_hard_bound_mu = np.mean(upper_bound[1:], axis=1)
upper_hard_bound_std = np.std(upper_bound[1:], axis=1)
ax.plot(t,upper_hard_bound_mu, label='Calculated upper bound', color='blue')
# ax.fill_between(t, upper_hard_bound_mu+upper_hard_bound_std, upper_hard_bound_mu-upper_hard_bound_std, facecolor='green', alpha=0.2)

ax.fill_between(t,upper_hard_bound_mu,lower_hard_bound_mu,facecolor='blue', alpha=0.1)
plt.legend()
plt.ylabel('Number of iterations for retraining')
plt.xlabel('Percentage of sparsity')
plt.savefig('lenet-5-random-1.png')
plt.show()

np.savez('lenet-5-random-1', num_non_zero_before_retraining = num_nonzero_before_retraining,
         num_non_zero_after_retraining=num_nonzero_after_retraining,
         lower_hard_bound=lower_hard_bound,
         upper_hard_bound=upper_hard_bound,
         gamma_series=gamma_series,
         test_acc_retraining=test_acc_list,
         num_iter_retrain=num_iter_list,
         grad_init=grad_init,
         test_loss_original=test_loss_original,
         test_acc_original=acc_final)
