# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.chdir('/home/andrej/ownCloud/sdc_p2/')



# Step 0

# Step 1:
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']



# Step 2:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_test[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)




# Step 3:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

plt.figure()
for i in range (20):
    plt.subplot(4,5, i+1)
    plt.imshow(X_train[i*1000+1])


plt.figure()
plt.subplot(1,2, 1)
plt.hist(y_train, bins= n_classes)
plt.title('Histrogram training-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)


plt.subplot(1,2, 2)
plt.hist(y_valid, bins= n_classes)
plt.title('Histrogram validation-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)

plt.figure() 
plt.subplot(1,1, 1)
plt.hist(y_test, bins= n_classes)
plt.title('Histrogram test-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)



# Step 4:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


import numpy as np



X_train= X_train - np.mean(X_train)
X_train = X_train / np.std(X_train)



plt.figure()
images_to_check = 6

for i in range(images_to_check):
       
    plt.subplot(images_to_check,3, i+1)
    plt.imshow(X_train[i*500+1])
    
    



   


# Step 5:
import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 128

# Step 6:

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 0: Convolutional. Input = 32x32x3. Output = 32x32x16.
    conv0_w = tf.Variable(tf.truncated_normal((5,5,3,128), mean= mu, stddev= sigma))
    conv0_b = tf.Variable(tf.zeros(128))
    conv0   = tf.nn.conv2d(x, conv0_w, strides=[1,1,1,1], padding='SAME') + conv0_b
    
    # TODO: Activation.
    conv0 = tf.nn.relu(conv0)

    drop0 = tf.nn.dropout(conv0, 0.8)
        
    # TODO: Layer 1: Convolutional. Input = 32x32x16. Output = 28x28x32.
    conv1_w = tf.Variable(tf.truncated_normal((5,5,128,64), mean= mu, stddev= sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1   = tf.nn.conv2d(drop0, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    drop1 = tf.nn.dropout(conv1, 0.7)
    
    # TODO: Layer 2: Convolutional. Input = 14x14x32. Output = 10x10x64.
    conv2_w = tf.Variable(tf.truncated_normal((5,5,64,32), mean= mu, stddev= sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(drop1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # TODO: Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    drop2 = tf.nn.dropout(conv2, 0.6)
    
    # TODO: Flatten. Input = 5x5x32. Output = 800.(because 5x5x32 = 1600)
    
    
    fc0 = flatten(drop2)
    
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(800,120), mean=mu, stddev=sigma) )
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    
    dropfc1 = tf.nn.dropout(fc1, 0.5)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(dropfc1, fc2_w) + fc2_b
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    dropfc2 = tf.nn.dropout(fc2, 0.5)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    fc3 = tf.matmul(dropfc2, fc3_w) + fc3_b
    
    logits = fc3
    
    return logits



# Step 7:

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Step 8:

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Step 9:
    
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Step 10:

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

training_acc= []
validation_acc= []
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy = evaluate(X_train, y_train)
        training_acc.append(training_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()
        
        validation_accuracy = evaluate(X_valid, y_valid)
        validation_acc.append(validation_accuracy)
        #print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    #saver.save(sess, './lenet')
    #print("Model saved")



t= np.linspace(1, EPOCHS, EPOCHS)
plt.title('Model. Attempt 5')
plt.plot(t, np.asarray(training_acc), 'b', label='Training Set')
plt.plot(t, np.asarray(validation_acc), 'r', label='Validation Set')
plt.legend(loc='lower rigth')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.grid(True)
plt.show()


# Step 11:
    
# Step 12:
























