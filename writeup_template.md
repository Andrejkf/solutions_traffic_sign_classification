# **Traffic Sign Recognition** 

## Writeup report for project 2, term 1.

This is the report for project 2, term 1. 

In this project a convolutional deep neural network model was used to clasiffy traffic signs. More specifically, the model was trained to classify images from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

It was used [anaconda](https://www.anaconda.com/) Python flavour (version 3.6.1), [scikit-learn](http://scikit-learn.org) (version 0.18.1), [TensorFlow GPU](https://www.tensorflow.org/) (version 1.3.0) and [OpenCV](https://opencv.org/releases.html) (version 3.4.0).

For the solution proposed, the next techniques were applied:

* [Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).
* [Data normalization](https://arxiv.org/pdf/1705.01809.pdf).
* [Shuffle training set](http://ieeexplore.ieee.org/document/8246726/?reload=true).
* [Batch Training](https://arxiv.org/abs/1711.00489).
* [Cross correlation](https://arxiv.org/abs/1309.5388).
* [Cross entropy](https://icml.cc/Conferences/2005/proceedings/papers/071_CrossEntropy_MannorEtAl.pdf).
* [Backpropagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf).
* [Stochastic gradient based optimization](https://arxiv.org/abs/1412.6980).

This is a non exclusive list of openCV functions I used:
* [cv2.createCLAHE()](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). Used for image contrast enhancement by applying adaptive histogram equalization.

* [cv2.resize()](https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html). Used for rescaling images to 32x32x3 size.

This is a non exhaustive list of Tensorflow functions I used:
* [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder). Used to feed in the input image in tensor representation.
* [tf.global_variables_initializer()](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer). Used to initialize all trainable variables.
* [tf.ConfigProto()](https://www.tensorflow.org/programmers_guide/using_gpu). Used to set up GPU memory usage upper boundary.
* [tf.train.AdamOptimizer()](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer). Used for stochastic gradient-based optimization.
* [tf.nn.softmax_cross_entropy_with_logits()](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits). Used to compute cross-entropy between *output network scores* and *expected claseification labels*. However this function is deprecated. Notice that I used this function because I am working with *TensorFlow version 1.3*, but, for *TensorFlow 1.6 and above* you may want to use [tf.nn.softmax_cross_entropy_with_logits_v2](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2) instead.
* [tf.nn.conv2d()](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). Used to compute the cross correlation between kernels and receptive fields.
* [tf.nn.max_pool()](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool). Used for [downsampling](https://web.stanford.edu/class/cs448f/lectures/2.2/Fast%20Filtering.pdf) each incoming feature map and reduce trainable variables at the same time.
* [tf.nn.relu()](https://www.tensorflow.org/api_docs/python/tf/nn/relu). Used as non-linear activation function, especifically a [Rectified Linear Unit](https://arxiv.org/abs/1611.01491), for model solution proposed.
* [tf.Session()](https://www.tensorflow.org/programmers_guide/graphs). Used to run tensor operations on the static computational graph.
* [tf.train.Saver()](https://www.tensorflow.org/programmers_guide/saved_model). Used to save and restore model variables. Which in fact, retrieves values from the checkpoints using C,C++ libraries under the hood.
* [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout). Used as regularization by dropping out units (both hidden and visible) in the model using pseudo-random probability depending of [tf.set_random_seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) to form random seeds.

---
## Reflection
To make it easier to follow up this reading the most relevant information a some iamges are used on this readme file, however, you can find all step by step process with full explanation in datail and all coding lines on file *solution_Traffic_Sign_Classifier.ipynb*.


---
### Content of this repository
* A file named **solution_Traffic_Sign_Classifier.ipynb** with the step by step detailed explanation of solution for this project in a  Jupyter IPython Notebook.
* ### todo * A file named **solution_Traffic_Sign_Classifier.ipynb** with the step by step detailed explanation of solution for this project in a  Jupyter IPython Notebook.
* A file named **signnames.csv** with  mappings from the class id (integer) to the actual sign name. It is just a copy of the *signnames.csv* file provided in the project.
* A folder named **traffic-signs-data** with the images used for model training, validation and test.
* A folder named **checkpoints** with metadata information of trained model.  In case that you want to use it please read the file *solution_Traffic_Sign_Classifier.ipynb*
* A folder named **parameter_tunning** with 7 python files corresponding each model tested during architecture design, in case that you want to check it out.
* A folder named **new_images_original** with the 10 images downloaded from the web to test the model after trained.
* A folder named **new_images_resized** with the same 10 images downloaded from the web to test the model after trained, but resized to (32x32x3) each.
* A folder named **other_images** with additional images used to ilustrate the solution approach on this report.

---
## Project Description.
**Build a Traffic Sign Recognition Project**

The specific goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
Here I will consider the *rubric points* individually and describe how I addressed each point in my implementation.  

---
### Files submitted
* A [writeup](https://github.com/Andrejkf/solutions_traffic_sign_classification) report in markdown that corresponds to this particular file you are reading right now.
*  A file named [**solution_Traffic_Sign_Classifier.ipynb**](https://github.com/Andrejkf/solutions_traffic_sign_classification) with the step by step detailed explanation of solution aproach and code for this project in a  Jupyter IPython Notebook.
* ### todo * A file named **solution_Traffic_Sign_Classifier.ipynb** with an HTML output of the code.
* A folder named **parameter_tunning** with 7 python files corresponding each model tested during architecture design, in case that you want to check it out.
*  A full [repository](https://github.com/Andrejkf/solutions_traffic_sign_classification) with all files described in the *content of this repository* sections of this document.



### 1. Data Set Summary & Exploration

I used python numpy library to calculate summary statistics of the traffic
signs data set:

On this German Traffic Signs dataset we have:

@ The number of training examples is: 34799
@ The number of validation examples is: 4410.
@ The number of testing examples is: 12630
@ The shape of a traffic sign image is: (32, 32, 3)
@ The number of unique classes/labels in the data set is: 43.
#### Exploratory visualization on the dataset.

Here we visualize the German Traffic Signs Dataset using the pickled files. Especifically we are displaying:
* In this first figure: A sample of 20 images from the training set.
<br/>![alt text][image1]
* In the seccond figure: A histogram(with distribution of classes) for the training and the validation set.
<br/> ![alt text][image2]
* In the third image: A histogram(with distribution of classes) for the test set.
 <br/>![alt text][image3]

From the first figure (with 20 samples of training images) we can subjectively notice that there is a brightness difference among the sampled images. Also, in some of these, a high rate brightness variation is present (e.g. image located in third row, third column). This suggest brightness and contrast transfomation would be a good candidate for data preprocessing.

From data set histograms (training,validation and test) unbalanced data is present. So data augmentation could be a solution to increase model generalization. More specifically to increase the number of feature examples for the less common classes.

### 2. Design and Test a Model Architecture
#### Step 2.1: Data preprocessing
After data exploration the follwing observations were taken into account to preprocess the data:
    
* Original input pixel image values range as usually was (0,255).
* The brighness among images use to highly variable.
* Some images have shown abrupt changes in brightness locally (very variable brightness in a single image).
* Data set was unbalanced.

This is the selected preprocessing approach:
    
Initially, it was considered to use gray scale images to have a more homogeneous contrast variation among the whole images dataset.
However, the three chanel colors representation was kept and all data preprocessing was done channel by channel. The previous idea was just an attempt to keep as much amount of possible usefull information on input data that the network might use during the weights tunning process.
In addition, to reduce abrupt changes present in some images it was decided to used histogram equalization by image sub-sections instead of using a global histogram equalization ([Click here for further information](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)).
By the other hand, even if the input data set was unbalanced it was wanted to [challenge the model generalization ability with unbalanced information](http://ieeexplore.ieee.org/document/6677603/) and penalized overfitting by using a regularization technique. More specifically, weights dropout during the training process.

##### Step 2.1.1: Contrast Limited Adaptive Histogram Equalization (CLAHE).
[Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) was used to get a better contrast enhancement than the one gotten with global histogram equalization.

A sample of 5 images is shown. It contains the original images and the corresponding images after contrast enhacement is applied.

<br/>![alt text][image4]

##### Step 2.1.2: Shuffled data.
Training set **mean** and **standard deviation** are computed to get zero-mean unit variace input data.
So we had closer zero-mean unit-variance input data. Also, a sample of the images after normalization is displayed below:
<br/>![alt text][image5]

#### Step 2.2: Solution Aproach.
Used a modified version of [LeNet-5](http://yann.lecun.com/exdb/lenet/).

As starting reference [LeNet-5 from laboratory]((https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) and a [convolutional network proposed by Aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) were used.
Then was modified the architecture changing and/or adding network parameters across the whole model.

Very general model tunning process followed is described as follows:
    
1. Tested LeNet-5 from lab as starting point.
2. Incresed batch size to 256 and run 10 Epochs.
3. Added dropout layers (after each ReLU) with keep_probability = 0.5 and run 50 Epochs.
4. Increased Number of filters in convolutional layers as going deeper (starting with 16 and ending with 64 filters)
5. Decreased Number of filters in convolutional layers as going deeper (starting with 128 and ending with 64 filters).
6. Changed probability to keep parameters in convolutional layers.
7. Reduced number of filters and added extra dropout layers after pooling_layers and increased probability to keep weights on convolutional, pooling and fully connedted layers.

So, basically I started with a simple networks that overfits quickly the training set, increased the number of free paramenters in the network and run for many epochs, but then I included regularization in a simpler model to increase generalization. I also had noticed that the harder you regularize the more epochs in takes more epochs to train the network but improves generalization.















As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

---
Links
[content of this repository]: https://github.com/Andrejkf/solutions_traffic_sign_classification

---
[//]: # (Image References)

[image1]: ./other_images/report_1.png "Sample images from training dataset"
[image2]: ./other_images/report_2.png "training and validation datasets histograms"
[image3]: ./other_images/report_3.png "test set histogram"
[image4]: ./other_images/report_4.png "contrast enhacement"
[image5]: ./other_images/report_5.png "zero-mean unit-variance"
[image6]: ./other_images/report_5.png "zero-mean unit-variance"

[image7]: ./other_images/report_5.png "zero-mean unit-variance"

[image8]: ./other_images/report_5.png "zero-mean unit-variance"

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNTg1NzEwNzRdfQ==
-->