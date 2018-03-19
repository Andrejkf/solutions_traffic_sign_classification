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
    
X_trainr, y_trainr = train['features'], train['labels']
X_validr, y_validr = valid['features'], valid['labels']
X_testr, y_testr = test['features'], test['labels']



# Step 2:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_trainr)

# TODO: Number of validation examples
n_validation = len(X_validr)

# TODO: Number of testing examples.
n_test = len(X_testr)

# TODO: What's the shape of an traffic sign image?
image_shape = X_testr[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_trainr))

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
    plt.imshow(X_trainr[i*1000+1])


plt.figure()
plt.subplot(1,2, 1)
plt.hist(y_trainr, bins= n_classes)
plt.title('Histrogram training-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)


plt.subplot(1,2, 2)
plt.hist(y_validr, bins= n_classes)
plt.title('Histrogram validation-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)

plt.figure() 
plt.subplot(1,1, 1)
plt.hist(y_testr, bins= n_classes)
plt.title('Histrogram test-set')
plt.xlabel('Image labels')
plt.ylabel('Number of images')
plt.grid(True)





# Step 4:





### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# convert to gray scale
import numpy as np

X_trainc = np.copy(X_trainr)
y_trainc = np.copy(y_trainr)

X_validc = np.copy(X_validr)
y_validc = np.copy(y_validr)

X_testc = np.copy(X_testr)
y_testc = np.copy(y_testr)


import cv2
import matplotlib.pyplot as plt


X_train = np.copy(X_trainc)
y_train = np.copy(y_trainc)

X_valid = np.copy(X_validc)
y_valid = np.copy(y_validc)

X_test = np.copy(X_testr)
y_test = np.copy(y_testc)



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


X_train[:][:,:,0].shape
X_train[0][:,:,0].shape

X_train[0][:,:,0] = clahe.apply(X_train[0][:,:,0])
X_train[0][:,:,1] = clahe.apply(X_train[0][:,:,1])
X_train[0][:,:,2] = clahe.apply(X_train[0][:,:,2])

for i in range(n_train):
    X_train[i][:,:,0] = clahe.apply(X_train[i][:,:,0])
    X_train[i][:,:,1] = clahe.apply(X_train[i][:,:,1])
    X_train[i][:,:,2] = clahe.apply(X_train[i][:,:,2])


for i in range(n_validation):
    X_valid[i][:,:,0] = clahe.apply(X_valid[i][:,:,0])
    X_valid[i][:,:,1] = clahe.apply(X_valid[i][:,:,1])
    X_valid[i][:,:,2] = clahe.apply(X_valid[i][:,:,2])


for i in range(n_test):
    X_test[i][:,:,0] = clahe.apply(X_test[i][:,:,0])
    X_test[i][:,:,1] = clahe.apply(X_test[i][:,:,1])
    X_test[i][:,:,2] = clahe.apply(X_test[i][:,:,2])





plt.figure()
images_to_check = 6
pointer = 0
for i in range(images_to_check):
    
    
    plt.subplot(images_to_check,2, pointer+1)
    plt.imshow(X_trainc[i*500+1])
    
    plt.subplot(images_to_check,2, pointer+2)
    plt.imshow(X_train[i*500+1])
    
    pointer +=2





############

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


import numpy as np



X_train= X_train - np.mean(X_train)
X_train = X_train / np.std(X_train)

X_valid = X_valid - np.mean(X_valid)
X_valid = X_valid /np.std(X_valid)

X_test= X_test - np.mean(X_test)
X_test = X_test / np.std(X_test)


plt.figure()
images_to_check = 6

for i in range(images_to_check):
       
    plt.subplot(images_to_check,3, i+1)
    plt.imshow(X_train[i*500+1])
    
    






# Step 5:
import tensorflow as tf

EPOCHS = 40
BATCH_SIZE = 256

# Step 6:

from tensorflow.contrib.layers import flatten

tf.reset_default_graph()
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal((5,5,3,6), mean= mu, stddev= sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    
    drop1a = tf.nn.dropout(conv1, keep_prob=0.9)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(drop1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    drop1b = tf.nn.dropout(pool1, keep_prob=0.9)


    # TODO: Layer 2: Convolutional. Input= 14x14x16. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal((5,5,6,16), mean= mu, stddev= sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(drop1b, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
    
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    drop2a = tf.nn.dropout(conv2, keep_prob=0.9)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(drop2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    
    drop2b = tf.nn.dropout(pool2, keep_prob=0.9)
    
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.(because 5x5x16 = 400)
    fc0 = flatten(drop2b)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma) )
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    
    dropfc1 = tf.nn.dropout(fc1, keep_prob=0.9)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(dropfc1, fc2_w) + fc2_b
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    dropfc2 = tf.nn.dropout(fc2, keep_prob=0.9)
    
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    
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
config.gpu_options.per_process_gpu_memory_fraction = 0.9

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
        
    saver.save(sess, './checkpoints/lenet')
    print("Model saved")



t= np.linspace(1, EPOCHS, EPOCHS)
plt.title('Model. Final attempt')
plt.plot( np.asarray(training_acc), 'b', label='Training Set')
plt.plot( np.asarray(validation_acc), 'r', label='Validation Set')
plt.legend(loc='lower rigth')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.grid(True)
plt.show()






# Step 11:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    


# Step 12:


import matplotlib.pyplot as plt


def read_image(path_image):
    '''
    Reads in image and used matplotlib.pyplot to display it
    
    path_image: use full path for image. example: /home/andrej/owncloud/images/1_1.jpeg
    '''
    import matplotlib.pyplot as plt
    
    a = plt.imread(path_image)
    print('Image dimensions:', a.shape)
    plt.imshow(a)    
    return a


def crop_image(path_in_image, x1,x2,x3,x4, path_out_image, flag):
    '''
    crops images
    path_in_image: use full path for incoming image. example: /home/andrej/owncloud/images/1_1.jpeg
    path_out_image: use full path for outgoing image. example: /home/andrej/owncloud/out_images/1_1_resized.jpeg
    x1,x2,x3,x4: are boundaries to crop image
    '''
    import cv2
    import matplotlib.pyplot as plt
    
    a = read_image(path_in_image)
    
    a_cropped = a[x1:x2, x3:x4]
        
    a_resized = cv2.resize(a_cropped, (32,32))
    print('resized image size:',a_resized.shape)
        
    if flag == 0:
        plt.imshow(a)
    elif flag == 1:
        plt.imshow(a_cropped)
    elif flag == 2:
        plt.imshow(a_resized)
    else:
        None
    
    plt.imsave(path_out_image, a_resized)
    
    return print('resized image and saved in path:', path_out_image)
    

crop_image('./new_images_original/2_1.ppm', 0,45,0,44, './new_images_resized/2_1_resized.png',2)
crop_image('./new_images_original/9_1.jpeg', 25,220,35,275, './new_images_resized/9_1_resized.png',2)
crop_image('./new_images_original/14_1.jpeg', 15,180,45,215, './new_images_resized/14_1_resized.png',2)
crop_image('./new_images_original/14_2.ppm', 0,43,0,42, './new_images_resized/14_2_resized.png',2)
crop_image('./new_images_original/20_1_flipped.jpeg', 45,190,15,165, './new_images_resized/19_1_resized.png',2)
crop_image('./new_images_original/20_1.jpeg', 45,190,15,165, './new_images_resized/20_1_resized.png',2)
crop_image('./new_images_original/25_1.ppm', 0,49,0,51, './new_images_resized/25_1_resized.png',2)
crop_image('./new_images_original/27_1.jpg', 35,345,15,340, './new_images_resized/27_1_resized.png',2)
crop_image('./new_images_original/29_1.ppm', 0,31,0,33, './new_images_resized/29_1_resized.png',2)
crop_image('./new_images_original/35_1.ppm', 0,51,0,53, './new_images_resized/35_1_resized.png',2)



   
in_images_names=['2_1_resized.png', '9_1_resized.png', '14_1_resized.png', '14_2_resized.png', '19_1_resized.png', '20_1_resized.png','25_1_resized.png', '27_1_resized.png', '29_1_resized.png', '35_1_resized.png' ]



X_newr = []
for i in in_images_names:
    #print(i)
    X_newr.append (cv2.imread('./new_images_resized/'+ i)[:,:,:3] )

X_new = np.asarray( X_newr )
X_new.shape





for i in range(X_new.shape[0]):
    X_new[i][:,:,0] = clahe.apply(X_new[i][:,:,0])
    X_new[i][:,:,1] = clahe.apply(X_new[i][:,:,1])
    X_new[i][:,:,2] = clahe.apply(X_new[i][:,:,2])

tmean = np.mean(X_trainr)
tstd = np.std(X_trainr- tmean)


plt.imshow(X_new[0])

X_new = X_new - tmean
np.mean(X_new)
X_new = X_new/tstd
np.std(X_new)








config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9


with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))

    new_predicted = sess.run( tf.argmax(logits, 1), feed_dict={x:X_new })
    print(new_predicted)

# Step 13
    
    
aux = np.reshape(X_new[3], (1,32,32,3))


def get_top_five(in_image):
    
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
        
        softmax = tf.nn.softmax(logits)
        top_prob = tf.nn.top_k(softmax, k =5 )
        top_five = sess.run( top_prob, feed_dict={x: in_image })
        #print(top_five)
    return top_five

aux2 = get_top_five(aux)

# step 14


def plot_5_prob(top_k, path_in_image, image_label):
    '''
    Plots top 5 logits probabilities in new images
    
    top_k: Vector with top predictions
    path_in_image: (str). Full path for image used for testing the model. Example: './new_images_resized/2_1_resized.png'
    image_label: (str). Label number for the class the image belongs to. Example: '2'
    '''
    certainty = np.asarray(top_k[0][0])
    predValues= list(map(unicode,list(top_k[1][0])))
    y_pos = np.arange(len(certainty))
    
    inImg= plt.imread(path_in_image)
    
    im1 = plt.subplot(1,2,1)
    plt.imshow(inImg)
    plt.title('Expected label: '+str(image_label))


    im2 = plt.subplot(1,2,2)
    im2.set_yticks(y_pos+0.1)
    im2.set_yticklabels(predValues)
    im2.barh(y_pos, certainty)
    im2.invert_yaxis()    

    plt.title('Top 5 probabilities predicted.')

    plt.show()


   
plot_5_prob(aux2, './new_images_resized/2_1_resized.png', '2')




# step 15

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={ x: image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            


            

stimulus_image= (X_trainr[31895]-tmean)/tstd

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))

    #for var in tf.trainable_variables():
    #    print(var)
    
    layer = sess.graph.get_tensor_by_name( 'Variable_2:0')
    outputFeatureMap(image_input=np.reshape(stimulus_image, (1,32,32,3)), tf_activation=layer)


plt.imshow(X_trainr[31895])













