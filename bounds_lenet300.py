'''
Author: Soumya Sara John
Requires tensorfloe == 1.15
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
import matplotlib.pyplot as plt
from random import randint

start_time = time.time()
'''
Input the data
'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
Define parameters
'''
learning_rate = 0.01
num_steps = 500
batch_size = 128
display_size = 100

n_hidden1 = 300
n_hidden2 = 100
num_input = 28*28
num_classes = 10

'''
Define the variables
'''
X = tf.placeholder("float",[None,num_input])
Y = tf.placeholder("float",[None,num_classes])

weights = {
    'h1' : tf.Variable(tf.random_normal([num_input,n_hidden1],mean = 0.0, stddev = 0.1)),
    'h2' : tf.Variable(tf.random_normal([n_hidden1,n_hidden2],mean = 0.0, stddev = 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden2,num_classes],mean = 0.0, stddev = 0.1))
}
weight = tf.Variable(tf.random_normal([num_input,num_classes],mean = 0.0, stddev = 0.1))
bias = tf.Variable(tf.random_normal([num_classes],mean = 0.0, stddev = 0.1))

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden1],mean = 0.0, stddev = 0.1)),
    'b2' : tf.Variable(tf.random_normal([n_hidden2],mean = 0.0, stddev = 0.1)),
    'out': tf.Variable(tf.random_normal([num_classes],mean = 0.0, stddev = 0.1))
}

'''
Define the layers of the network
'''
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X,weights['h1']),biases['b1']))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
out_layer = tf.add(tf.matmul(layer_2,weights['out']),biases['out'])

logits = out_layer
predictions = tf.nn.softmax(logits)

'''
Define train operations
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
grads = optimizer.compute_gradients(loss, var_list = weights)
grads_ = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_)
# train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(predictions,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

'''
Functions for pruning
Doing random sparsing on the weights and biases
Choosing random connections to be pruned
'''
def prune(percentage_of_pruning, weight):
  weight_sparse = weight.copy()
  percentage_of_pruning = percentage_of_pruning*weight.shape[0]*weight.shape[1]
  for s in range(int(percentage_of_pruning)):
    i = randint(0,weight.shape[0]-1)
    j = randint(0,weight.shape[1]-1)
    weight_sparse[i][j] = 0
  no_nonzero_before = np.count_nonzero(weight)
  no_nonzero_after = np.count_nonzero(weight_sparse)
  # print('no_nonzero_after', no_nonzero_after)
  # print('no_nonzero_before',no_nonzero_before)
  return weight_sparse

def k_means(n_clusters, layer_out):
    estimator = KMeans(n_clusters=n_clusters, init = 'random', n_init = 70)
    cluster_ = estimator.fit(layer_out)
    labels = cluster_.labels_
    clusters = []
    for i in range(n_clusters):
        loc = list(np.where(labels == i))[0]
        if loc.size > 0:
            clusters.append(loc)
    clusters = np.array(clusters)
    return clusters

def find_AB(hu,hv,a_initial, b_initial, conv, max_iter):
    alpha = a_initial
    beta = b_initial
    residue = hu - alpha*hv - beta
    residue_ = np.dot(residue,residue.T)
    t = 0
    while(residue_>conv and t<max_iter):
        alpha = (np.dot((hu-beta),hv.T))/(np.dot(hv,hv.T)+0.01)
        beta = hu - alpha*hv
        residue = hu - alpha*hv - beta
        residue_ = np.dot(residue,residue.T)
#        print(residue_)
        t = t + 1
    return alpha, beta

def correlation_based_fc(clusters, wp, wn, b, layer_out):
    wp_new = np.zeros((wp.shape))
    wn_new = np.zeros((wn.shape))
    b_new = np.zeros((b.shape[0]))
    nodes = []
    for i in range(np.array(clusters).shape[0]):
        bias = []
        for j in range(clusters[i].shape[0]):
            bias.append(np.abs(b[clusters[i][j]]))
        l = np.argmax(bias)
        wp_new[:,clusters[i][l]] = wp[:,clusters[i][l]]
        wn_new[clusters[i][l],:] = wn[clusters[i][l],:]
        b_new[clusters[i][l]] = b[clusters[i][l]]
        nodes.append(clusters[i][l])
        p = 0
        for k in clusters[i]:
            alpha_,beta_ = find_AB(layer_out[k,:],layer_out[clusters[i][l],:], a_initial = 1, b_initial = np.zeros((layer_out[k,:].shape)),conv = 1e-3, max_iter = 10)
            wn_new[clusters[i][l],:] = wn_new[clusters[i][l],:] + alpha_*wn[k,:]
            b_new[clusters[i][l]] = b_new[clusters[i][l]] + np.average(beta_)*b[k]
            p = p + 1
    return wp_new, wn_new, b_new, nodes

'''
Pruning and retraining the network
'''
init = tf.global_variables_initializer()
t_init = num_steps
epsilon = 0.01
with tf.Session() as sess:
    sess.run(init)
    vars_ = tf.trainable_variables()
    g_init = sess.run(grads, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}) # size(3,2) and gradients are the 0th ones of each row.
    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X:batch_x,Y:batch_y})
        if step % display_size == 0 or step == 1:
            loss1,acc = sess.run([loss,accuracy],feed_dict={X:batch_x,Y:batch_y})
            print("Step:{},Minibatch loss = {} ,training accuracy={}".format(step,loss1,acc))
    print("Optimization over")
    end_time = time.time()
    print("Time taken for training = {}s".format(end_time-start_time))
    test_loss_original, test_acc_original = sess.run([loss,accuracy],feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
    print("Testing accuracy:",test_acc_original, "Testing loss: ", test_loss_original)
    end_time = time.time()
    print("Time taken for testing:{}s".format(time.time() - end_time))
    wh1 = weights['h1'].eval()
    wh2 = weights['h2'].eval()
    wout = weights['out'].eval()
    bh1 = biases['b1'].eval()
    bh2 = biases['b2'].eval()
    bout = biases['out'].eval()

    grad_init = 0
    for gi in g_init:
      grad_init = grad_init + np.square(np.linalg.norm(gi[0]))
    pr = 10
    itr = 50
    num_non_zero_before_retraining = np.zeros((pr, itr))
    num_non_zero_after_retraining = np.zeros((pr, itr))
    lower_hard_bound = np.zeros((pr,itr))
    upper_hard_bound = np.zeros((pr,itr))
    gamma_series = np.zeros((pr,itr))
    test_acc_pruning = np.zeros((pr,itr))
    test_loss_pruning = np.zeros((pr,itr))
    test_acc_retraining = np.zeros((pr,itr))
    test_loss_retraining = np.zeros((pr,itr))
    num_iter_retrain = np.zeros((pr,itr))
    for p in range(1,pr):
      pop = p/10
      print(p)
      for i in range(itr):
        wh1_sparse = prune(pop, wh1)
        wh2_sparse = prune(pop, wh2)
        wout_sparse = prune(pop, wout)
        sess.run(init)
        sess.run(tf.assign(weights['h1'], wh1_sparse))
        sess.run(tf.assign(weights['h2'], wh2_sparse))
        sess.run(tf.assign(weights['out'], wout_sparse))
        sess.run(tf.assign(biases['b1'], bh1))
        sess.run(tf.assign(biases['b2'], bh2))
        sess.run(tf.assign(biases['out'], bout))
        g_sparse = sess.run(grads, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}) # size(3,2) and gradients are the 0th ones of each row.
        delta_w = np.square(np.linalg.norm(wh1_sparse - wh1)) + np.square(np.linalg.norm(wh2_sparse - wh2)) + np.square(np.linalg.norm(wout_sparse - wout))
        grad_sparse = 0
        for gs in g_sparse:
          grad_sparse = grad_sparse + np.square(np.linalg.norm(gs[0]))
        lower_hard_bound[p][i] = grad_sparse/grad_init
        upper_hard_bound[p][i] = delta_w/(2*epsilon*learning_rate)
        gamma_series[p][i] = delta_w*grad_init/(2*epsilon*learning_rate*num_steps*grad_sparse)

        '''
        Retraining the sparse network
        '''
        num_non_zero_before_retraining[p][i] = np.count_nonzero(wh1_sparse) + np.count_nonzero(wh2_sparse) + np.count_nonzero(wout_sparse)
        test_loss, test_acc = sess.run([loss,accuracy], feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
        test_loss_pruning[p][i] = test_loss
        test_acc_pruning[p][i] = test_acc
        num_iter = 0
        while(np.abs(test_loss - test_loss_original) > epsilon):
          num_iter = num_iter + 1
          batch_x,batch_y = mnist.train.next_batch(batch_size)
          sess.run(train_op, feed_dict={X:batch_x,Y:batch_y})
          loss1,acc = sess.run([loss,accuracy],feed_dict={X:batch_x,Y:batch_y})
          test_loss = sess.run(loss,feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
        test_acc = sess.run(accuracy,feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
        test_acc_retraining[p][i] = test_acc
        test_loss_pruning[p][i] = test_loss
        wh1_sparse = weights['h1'].eval()
        wh2_sparse = weights['h2'].eval()
        wout_sparse = weights['out'].eval()
        num_non_zero_after_retraining[p][i] = np.count_nonzero(wh1_sparse) + np.count_nonzero(wh2_sparse) + np.count_nonzero(wout_sparse)
        num_iter_retrain[p][i] = num_iter
        print('p: ',p,' i: ',i, ' Final test accuracy: ', test_acc)
        print('num_iter: ', num_iter)

'''
Calculate the bounds
'''
f1 = np.zeros((pr,1))
f2 = np.zeros((pr,1))
for i in range(1,pr):
  f1[i] = i/10
  f2[i] = (1e-10)*(i/10)
lower_bound = np.multiply(f1, lower_hard_bound)
gammas = np.multiply(f2, gamma_series)
upper_bound = np.multiply(gammas, upper_hard_bound)

'''
Plot and save the required values
'''
fig, ax = plt.subplots(1)
t = [10,20,30,40,50,60,70,80,90]
num_iter_retrain_mu = np.mean(num_iter_retrain[1:], axis=1)
num_iter_retrain_std = np.std(num_iter_retrain[1:], axis=1)
ax.plot(t,num_iter_retrain_mu, label='Experimentally obtained', color='red')
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
plt.savefig('lenet-300-random-1.png')
plt.show()

np.savez('lenet-300-random-1', num_non_zero_before_retraining = num_non_zero_before_retraining,
         num_non_zero_after_retraining=num_non_zero_after_retraining,
         lower_hard_bound=lower_hard_bound,
         upper_hard_bound=upper_hard_bound,
         gamma_series=gamma_series,
         test_acc_pruning=test_acc_pruning,
         test_loss_pruning=test_loss_pruning,
         test_acc_retraining=test_acc_retraining,
         test_loss_retraining=test_loss_retraining,
         num_iter_retrain=num_iter_retrain,
         grad_init=grad_init,
         test_loss_original=test_loss_original,
         test_acc_original=test_acc_original)
