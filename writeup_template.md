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

As starting reference **LeNet-5 from laboratory** and a [convolutional network proposed by Aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) were used.
Then the architecture was modified changing and/or adding network parameters across the whole model.

Very general model tunning process followed is described as follows:
    
1. Tested LeNet-5 from lab as starting point.
2. Incresed batch size to 256 and run 10 Epochs.
3. Added dropout layers (after each ReLU) with keep_probability = 0.5 and run 50 Epochs.
4. Increased Number of filters in convolutional layers as going deeper (starting with 16 and ending with 64 filters)
5. Decreased Number of filters in convolutional layers as going deeper (starting with 128 and ending with 64 filters).
6. Changed probability to keep parameters in convolutional layers.
7. Reduced number of filters and added extra dropout layers after pooling_layers and increased probability to keep weights on convolutional, pooling and fully connedted layers.

So, basically I started with a simple networks that overfits quickly the training set, increased the number of free paramenters in the network and run for many epochs, but then I included regularization in a simpler model to increase generalization. I also had noticed that the harder you regularize the more epochs in takes more epochs to train the network but improves generalization.
Below are shown images for tunning process from step 2 to step7. On the last image is displayed the selected model without the final hyper parameters tunning:
<br/>![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] 

#### Step 2.3: Model Architecture.

Finally, the main goal was to get a model as simple as possible. For that reason, the following architecture was chosen:

<table class="tg">
  <tr>
    <th class="tg-031e">Layer name<br></th>
    <th class="tg-031e">Layer type<br></th>
    <th class="tg-031e">Parameters/Description</th>
  </tr>
  <tr>
    <td class="tg-031e">conv1</td>
    <td class="tg-031e">Convolution</td>
    <td class="tg-031e">Filters: 6 (5x5)<br>Pad: 0<br>At stride: 1<br>Activation: ReLU<br>Input: (32x32x3)<br>Output:(28x28x6)<br></td>
  </tr>
  <tr>
    <td class="tg-031e">drop1a</td>
    <td class="tg-031e">Dropout</td>
    <td class="tg-031e">Keep Prob: 0.8<br></td>
  </tr>
  <tr>
    <td class="tg-031e">pool1</td>
    <td class="tg-031e">Pooling</td>
    <td class="tg-031e">Kernels size: (2x2)<br>Pad: 0<br>At stride: 2<br>Input: (28x28x6)<br>Output: (14x14x6)<br></td>
  </tr>
  <tr>
    <td class="tg-yw4l">drop1b</td>
    <td class="tg-yw4l">Dropout</td>
    <td class="tg-yw4l">Keep Prob: 0.9</td>
  </tr>
  <tr>
    <td class="tg-yw4l">conv2</td>
    <td class="tg-yw4l">Convolution</td>
    <td class="tg-yw4l">Filters: 16 (5x5)<br>Pad: 0<br>At stride: 1<br>Activation: ReLU<br>Input: (14x14x16)<br>Output:(10x10x16)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">drop2a</td>
    <td class="tg-yw4l">Dropout</td>
    <td class="tg-yw4l">Keep Prob: 0.9</td>
  </tr>
  <tr>
    <td class="tg-yw4l">pool2</td>
    <td class="tg-yw4l">Pooling</td>
    <td class="tg-yw4l">Kernels size: (2x2)<br>Pad: 0<br>At stride: 2<br>Input: (10x10x16)<br>Output: (5x5x16)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">drop2b</td>
    <td class="tg-yw4l">Dropout</td>
    <td class="tg-yw4l">Keep Prob: 0.9</td>
  </tr>
  <tr>
    <td class="tg-yw4l">fc0</td>
    <td class="tg-yw4l">Flatten</td>
    <td class="tg-yw4l">Input: (5x5x16)<br>Output: 400 Neurons<br></td>
  </tr>
  <tr>
    <td class="tg-yw4l">fc1</td>
    <td class="tg-yw4l">Fully Connected<br></td>
    <td class="tg-yw4l">Input: 400 Neurons<br>Output: 120 Neurons</td>
  </tr>
  <tr>
    <td class="tg-yw4l">dropfc1</td>
    <td class="tg-yw4l">Dropout</td>
    <td class="tg-yw4l">Keep Prob: 0.9</td>
  </tr>
  <tr>
    <td class="tg-yw4l">fc2</td>
    <td class="tg-yw4l">Fully Connected</td>
    <td class="tg-yw4l">Input: 120 Neurons<br>Output: 84 Neurons</td>
  </tr>
  <tr>
    <td class="tg-yw4l">dropfc2</td>
    <td class="tg-yw4l">Dropout</td>
    <td class="tg-yw4l">Keep Prob: 0.9</td>
  </tr>
  <tr>
    <td class="tg-yw4l">fc3</td>
    <td class="tg-yw4l">Fully Connected</td>
    <td class="tg-yw4l">Input: 84 Neurons<br>Output: 43 Neurons (Class Scores)<br></td>
  </tr>
</table>


With the following hyper parameters values:
* Batch size = 256 
* Training Epochs = 50 (Please check last plot where the model has performace above 93% in validation set after epoch number 25 aproximately ). Was preferred to stop model at 50 epochs to avoid over-training.
* Learning rate : 0.001 (I did not used decaying learning rate because [Adam optimizer computes individual adaptive learning rate](https://arxiv.org/pdf/1412.6980.pdf).
* Dropout probability = 0.1 ( Keep probability 0.9).



##### Description about how the final model was trained.
The following considerations were made during training stage:

* To train the model the initial batch size was 128 but changed to 256 with out any GPU/CPU/RAM memory problems. During initial hyper parameters tunning just the training and validation sets were used.
* The training set was shuffled every epoch.
* The first one to tune was the learning rate, using initially values close to 0.005 with a high error, then used low values close to 0.0005 but the training accuracy did not imporved after many epochs. So a value of 0.001 was set up.
* Then the number of epochs was set to a high number like 100 and 150 and compared training vs validation accuracy.
* After noticed a high tendency of the network to overfit training data, the probability of droping out weights was set to 0.5, 0.6, 0.7,0.8 and 0.9 and noticed that better performance was with low dropout levels, that is, with high probability of keeping the same weights. So a dropout=0.1 (keep probabilty = 0.9) was chosen.
* Known that the networks looks for overfitting data, an early stop training of 50 epochs was selected to avoid it.
* Model predictions were followed by softmax probabilities.
* Then cross-entropy between predicted labels and expected label values was computed.
* The objective function selected was the mean error rate between logits and labels.
*After training was finished, model parameters where saved for further model performance evaluation.

**Note**: The selected optimization algoritm was [Adam](https://arxiv.org/pdf/1412.6980.pdf).

The final model results at epoch 50 were:
-   training set accuracy of 0.988 = 98.8%
-   validation set accuracy of 0.939 = 939%
-   test set accuracy of 0.93 = 93%

Bellow the final model setup training and validation accuracy are displayed:
<br/>![alt text][image12]

It is important to note that convolutional layers were selected because cross-correlations done at receptive fields generate activation maps that detect image patterns. [Read this paper for further information](https://arxiv.org/abs/1311.2901).

Also, dropout layers can help to distribute knowledge across the network. [Read this paper for further information](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).


### Step 3: Test the Model on New Images

Five (5) pictures of German traffic signs were downloaded from the web using google browser and other five (5) where downloaded from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). Then the pretrained model was used to predict the traffic sign type.

#### Step3.1: Load and Output the Images
A total of 10 new images were used for further testing purposes. Some of them where selected to be the less common classes images in the dataset to test model robustness.


The first 5 downloaded images are show below.

<br/>![alt text][image13] ![alt text][image17]
![alt text][image14] ![alt text][image15] 
![alt text][image16] 

* Image 1, with label: 9,No passing. Was selected because is not in he common position of the images from training set. So even if that image class was very common on the training set given the fact that data augmentation was not done is expected to misclasify this image.
* Image 2,  with label: 27,Pedestrians. Was selected because in the read is mandatory to detect any information related with pedestrians. In this case, kids from kindergarten.
* Image 3,  with label: 14,Stop. Was taken because semantically is one of the most important traffic signs on the road. The image was selected with high brigtness to test the importance of including contrast enhancement during preprocessing images. This one is expected to be correctly classified.
* Image 4, with label: 20,Dangerous curve to the right. Was chosen for being one with less common class type in the training set. This image is expected to be misclassifed.
* Image 5, with label: 19,Dangerous curve to the left. Was chosen also for being one of the commonless in the training set. Is the flipped verion of the downloaded image with label 20,Dangerous curve to the right. This image is expected to be misclasified too.

The other 5 images were downloaded from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) and links to view then are shown below. They were chosen to be very difficult for me, as a human, to clasify them.
<br/> ![alt text][image18] ![alt text][image19]
![alt text][image20] ![alt text][image21] 
![alt text][image22] 

* Image 6, with label: 2,Speed limit (50km/h). Was selected because the image was taken while on the road and some pixels seem to be shifted. It is one of the most common images on the data set so it is expected that it should be correctly classified.
* Image 7, with label: 14,Stop. Was selected because exhibits a very variable level of brigthness among the different pixels. A stop signal is one of the most important signals to be detected.
* Image 8, with label: 25,Road work. Was selected because is one of the classes with the low level number of examples in the training set.
* Image 9,  with label:  29,Bicycles crossing. Was chosen becuase as previously mentioned, everything related to pedestrians or human people involved is a priority. Also the image is not easy to identify by a human eye given the fact it has a high brightness level.
* Image 10, with label: 35,Ahead only. Was selected as an attempt to push the network to misclasify it.

#### Step 3.2: Performance on new images
Here are the results of the prediction:
<br/> expected\_labels\_new:  \[ 2  9 14 14 19 20 25 27 29 35\]
<br/> new_predicted values: \[ 2 41 14 18 40 41 25 24 29  9\]


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 2,Speed limit (50km/h)      		| 2,Speed limit (50km/h)   									| 
| 9,No passing     			| 41,End of no passing 										|
| 14,Stop				| 14,Stop											|
| 14,Stop      		| 18,General caution Road					 				|
| 19,Dangerous curve to the left			| 40,Roundabout mandatory Road  
| 20,Dangerous curve to the right      		| 41,End of no passing sign   									| 
| 25,Road work     			| 25,Road work 										|
| 27,Pedestrians				| 24,Road narrows on the right											|
| 29,Bicycles crossing      		| 29,Bicycles crossing Road					 				|
| 35,Ahead only			| 9,No passing  


As result the model had 4/10 correct predicted classes, that is, 40% of accuracy in the new images.

From the test set the minimum accuracy performance obtained was 93% in a total  of 12630 examples. By the other hand, the accuracy on the mini data sample set from the downloaded images after running just one time was 4/10 = 40% in a total of 10 images.

Notice that the amount of image examples from test set is bigger that the amount of examples in the mini data set of new downloaded images in a ratio of 12360/10 = 1236.

Reader migth be tempted to make a performance comparison between *test* and *new images* data sets. In fact, you can be tempted to concluded that *the probability that **the accuracy of 93% obtained in the test set** to characterize the network performance is higher than the probability that **the accuracy of 40% obtained on the new mini set***.
However, that asumption might not be true and is preferred not to generate affirmations before studying better the test data set and the new images mini data set.

In addition, author does not compromise to any conclusion about this attempt to compare *test_set* with *new_imags_mini_dataset*.

#### Step 3.3: Output Top 5 Softmax Probabilities For Each Image Found on the Web
The code for making predictions on my final model is located in the section **Step3.3** of the Ipython notebook.

In this subsection top softmax probabilities for each new image found on the web are displayed:

<br/> ![alt text][image23] ![alt text][image24]
![alt text][image25] ![alt text][image26] 
![alt text][image27] ![alt text][image28] ![alt text][image29]
![alt text][image30] ![alt text][image31] 
![alt text][image32] 

##### Model Certainty- Softmax Probabilities discussion.
1. Notice that the fact that the model predicts a class label with probability 1 does not necessarily implies that the clasification is correct, it might be wrong. 

    For example, in the image with *expected label:2* the softmax probability of being label as 'predicted_label:2' was 1 = 100%. However for the image with *expected label: 19* the softmax probability of being label as 'predicted_label:40' was 1, which is totally wrong.
    
    Also please notice that the amount of examples for class '2' is one of the most common in training set, but, the amount of examples for class '19' is one of the commonless.
    
2. Notice that sometimes the expected label number were not listed in the top 5 softmax probabilities, showing that the trained network has real trouble classifiying these images.

3. Notice also that the network was tested just one time with the *new mini data set of images downloaded from the web*. So you will have different performance everytime you run the test in this new mini data set.


### Step 4: Visualizing the Neural Network's State with Test Images.

On this section a visualization of the filters activation is done but not the one with the activation maps.

Withal, the author is working in extracting the feature maps from hidden layers just to check hidden layers networks response.
An stimulus image is used to get the filters response.
<br/> ![alt text][image33] 

And the Weigts for filters in Convolutional layer 2 are shown:
<br/> ![alt text][image34]

* It is interesting to notice how Filter2 and Filter3 show high number of weights. 
* Also, 

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




---
[//]: # (Image References)

[image1]: ./other_images/report_1.png "Sample images from training dataset"
[image2]: ./other_images/report_2.png "training and validation datasets histograms"
[image3]: ./other_images/report_3.png "test set histogram"
[image4]: ./other_images/report_4.png "contrast enhacement"
[image5]: ./other_images/report_5.png "zero-mean unit-variance"
[image6]: ./other_images/2.png "model 2"

[image7]: ./other_images/3.png "model 3"

[image8]: ./other_images/4.png "model 4"
[image9]: ./other_images/5.png "model 5"
[image10]: ./other_images/6.png "model 6"
[image11]: ./other_images/7.png "model 7"

[image12]: ./other_images/report_6.png "Final model trained"

[image13]: ./other_images/report_original_9_1.png  "Label: 9,No passing"
[image14]: ./other_images/report_original_14_1.png "Label: 14,Stop"
[image15]: ./other_images/report_original_20_1.png "Label: 20,Dangerous curve to the right."
[image16]: ./other_images/report_original_20_1_flipped.png "Flipped version of previous image"
[image17]: ./other_images/report_original_27_1.png "Label: 27,Pedestrians"


[image18]: ./other_images/report_original_2_1.png  "Label: 2,Speed limit (50km/h)"
[image19]: ./other_images/report_original_14_2.png "Label: 14,Stop"
[image20]: ./other_images/report_original_25_1.png "Label: 25,Road work"
[image21]: ./other_images/report_original_29_1.png "Label: 29,Bicycles crossing"
[image22]: ./other_images/report_original_35_1.png "Label: 35,Ahead only"


[image23]: ./other_images/report_11.png "sofmax img1"
[image24]: ./other_images/report_12.png "sofmax img2"
[image25]: ./other_images/report_13.png "sofmax img3"
[image26]: ./other_images/report_14.png "sofmax img4"
[image27]: ./other_images/report_15.png "sofmax img5"
[image28]: ./other_images/report_16.png "sofmax img6"
[image29]: ./other_images/report_17.png "sofmax img7"
[image30]: ./other_images/report_18.png "sofmax img8"
[image31]: ./other_images/report_19.png "sofmax img9"
[image32]: ./other_images/report_20.png "sofmax img10"



[image33]: ./other_images/report_41.png "stimuli1"
[image34]: ./other_images/report_42.png "Filters conv 2"

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAzNzM1MTA1N119
-->