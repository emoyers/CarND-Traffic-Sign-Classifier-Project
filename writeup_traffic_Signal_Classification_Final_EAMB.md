# **Traffic Sign Recognition**

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/histogram_1.jpg "Histogram original data set"
[image2]: ./report_images/grayscale.jpg "Grayscaling"
[image3]: ./report_images/rotation.jpg "Rotation"
[image4]: ./report_images/translation.jpg "tTranslation"
[image5]: ./report_images/brightness.jpg "Brightness"
[image6]: ./report_images/histogram_2.jpg "Histogram new data set"
[image7]: ./report_images/Sermanet_LeCunn_arq.jpg "Sermanet/LeCunn Arquitecture"
[image8]: ./report_images/lenet_arq.jpg "Lenet Arquitecture"
[image9]: ./new_test_images/image_8.jpg "Bumpy road"
[image10]: ./new_test_images/image_5.jpg "Right-of-way at the next intersection"
[image11]: ./new_test_images/image_7.jpg "No entry"
[image12]: ./new_test_images/image_9.jpg "General Caution"
[image13]: ./new_test_images/image_1.jpg "Road Work"
[image14]: ./new_test_images/image_4.jpg "Speed limit (70km/h)"
[image15]: ./new_test_images/image_2.jpg "Stop"
[image16]: ./new_test_images/image_6.jpg "Speed limit (30km/h)"
[image17]: ./new_test_images/image_3.jpg "Yield"
[image18]: ./report_images/prediction.jpg "Predictions"
[image19]: ./report_images/step4.jpg "Feauture maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/emoyers/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 53850
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the **original** data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because for this case there is no difference if the image were processed in color or in grayscale since it not exist a set of signals that by changing the color the meaning will change. At least for this 43 classes example.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because Lenet arquitecture works better with image that have a normal distribution with a mean of 0.

I decided to generate additional data because the first data set has imbalance data distribution as you can see:

![alt text][image1]

To add more data to the the data set, I used the following techniques:

* Random rotation

![alt text][image3]

* Random transalation

![alt text][image4]

* Random brightness modification

![alt text][image5]

The difference between the original data set and the augmented data set is the following:

![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was based in Sermanet/LeCunn that I found in the [link](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and has the following layers:

![alt text][image7]

| Layer         		|  Description	        					|
|:-----------------:|:----------------------------------:|
| Input         		| 32x32x3 RGB image   			|
| Convolution 5x5 (1)  |   1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5 (2)  | 1x1 stride, valid padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Convolution 5x5 (3) | 1x1 stride, valid padding, outputs 1x1x400		|
| RELU					|												|
| Flatten	(1)	| inputs Convolution 5x5 (3), outputs 400   									|
| Flatten	(2) | inputs Convolution 5x5 (2), outputs 400   									|
| Concatenation			| inputs Flatten	(1) and  Flatten	(2), outputs 800		|
| Dropout     |                         |
| Fully Connected | outputs 43  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used **Softmax Cross Entropy**, then **Reduce Mean** and **Adam Optimizer** with **Batch size** equal to **100** , number of **Epoch** equal to **100**, **learning rate** equal to **0.0009**, **mu** equal to **0**, **sigma** equal to **0.1** and **Keep probability** for the **Dropout** equal to **0.5**.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **96.3**
* validation set accuracy of **96.3**
* test set accuracy of **94.9**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
**Was Lenet arquitecture**
![alt text][image8]
* What were some problems with the initial architecture? **It didn't reach the desired 93 value**
* How was the architecture adjusted and why was it adjusted? **That is why I chose to look for another one and I found Sermanet/LeCunn**
* How might a dropout layer help with creating a successful model? **Droput ot help the design to avoid over fitting.**

If a well known architecture was chosen:
* What architecture was chosen? **Sermanet/LeCun**
* Why did you believe it would be relevant to the traffic sign application? **It was already teste in traffic signals. See** [link](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? **The results with the new architecture for test images were 94.9.**



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14]
![alt text][image15] ![alt text][image16] ![alt text][image17]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Bumpy road		        |     Bumpy road	        					|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   					|
| No entry     			    | No entry 										|
| General Caution				| General Caution											|
| Road Work	      		  | Road Work					 				|
| Speed limit (70km/h)	| Speed limit (70km/h)      							|
| Stop				          | Yield											|
| Speed limit (30km/h)	| Speed limit (30km/h)					 				|
| Yield			            | Yield      							|

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.89%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 33th cell of the Ipython notebook.

Here is image showing the summary of the first 3 prediction of each image:

![alt text][image18]


### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The example here show how the neural network start learning the **triangle characteristic** of the **Bumpy road** traffic sign.

![alt text][image19]
