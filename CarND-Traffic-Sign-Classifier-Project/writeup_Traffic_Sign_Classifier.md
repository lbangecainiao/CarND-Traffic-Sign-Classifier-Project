# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/TrafficSign0.png "Traffic Sign 1"
[image5]: ./examples/TrafficSign1.png "Traffic Sign 2"
[image6]: ./examples/TrafficSign2.png "Traffic Sign 3"
[image7]: ./examples/TrafficSign3.png "Traffic Sign 4"
[image8]: ./examples/TrafficSign4.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows the images of every traffic class(43 in total) and its corresponding number. It could be checked using the signnames.csv file to verify its correctness.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color is not a necessary information for the traffic sign classifier. It's enough to use the grayscale plot to reduce the unnecessary input information.

Here is an example of a traffic sign image before and after grayscaling. 

![alt text][image2]

As a last step, I normalized the image data because it could guarantee a 0 expectation of the data and makes the optimizer easier to find the optimum solution.

I decided to generate additional data because the frequency of some traffic sign are too low.Augmenting the number of the traffic sign could increase the accuracy of the model.

To add more data to the the data set, I used the following techniques including adjusting the brightness of the image. Translate the image by a random pixel. Scaling the image by a random pixel. I didn't implement the random rotation of the traffic sign, since the rotation could modify the meaning of the traffic sign. For instance rotating the traffic sign "turning left" by 180 degree could modify the meaning to "turning right", which will cause a mistake in the training data set and thus should be avoided.

The difference between the original data set and the augmented data set is the following. 

![alt text][image3]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| DROPOUT					|keep probability 0.5												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| DROPOUT					|keep probability 0.5												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| FLATTEN					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Fully connected		| outputs 120.        									|
| RELU					|												|
| DROPOUT					|keep probability 0.5												|
| Fully connected		| outputs 84.        									|
| RELU					|												|
| DROPOUT					|keep probability 0.5												|
| Fully connected		| outputs 43.        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer.The adopted batch size is 128. The number of epochs of the last training is 40. However the results of the previous iteration of traings are saved and loaded before the new training iteration is started. Thus the actual number of EPOCHS is larger than 40. The learning rate in the last training is 0.0007; the initialization parameters of the weight coefficients are mu = 0 and sigma = 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 100%
* validation set accuracy of: 94.3% 
* test set accuracy of : 93.0%

If a well known architecture was chosen:
* What architecture was chosen?
  The LeNet network is chosen.
* Why did you believe it would be relevant to the traffic sign application?
  Since it could successfully identify the numerical number in the MNIST data. It could be possibly implemented in the traffic sign classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  The model could achieve a relative high performance with training accuracy as 100%, validation accuracy as 94.3% and test accuracy as 93.0%.
* Steps to achieve the validation accuracy:
  1.First I implement the LeNet network and the given dataset. I could only achieve a validation accuracy around 70%.
  2.I add the dropout layer after each RELU layer, and tried the data augmentation technique. It finally brought the validation accuracy to the 94.3%, and test accuracy to 93.0%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because the color is too dark to recognize even by human eye.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Turn right ahead					| Turn right ahead											|
| General caution	      		| General caution					 				|
| Speed limit (100km/h)			| Speed limit (100km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.0%.

*Several issues occured during the testing of the images found on the web
1.The size of the image is not exactly 32x32, which cannot be fitted into the neural network. It's necessary to resize the image.
2.The image might have 4 color channel(RGBA). It's necessary to conver the color channel into 3 color channel(RGB).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

We could notice that for the five traffic sign the neural network gave a high prediction probability almost equals to 1.00 except the second image. Observing the image I found out that the second traffic sign is too large and not completely shown in the image(some part are cut). While most of the images from the training set are complete. Thus this might be a possible reason causing the probability slightly lower.

| Probability 			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| Priority road   									| 
| 0.93      			| Right-of-way at the next intersection 										|
| 1.00					| Turn right ahead											|
| 1.00	      		| General caution					 				|
| 1.00			| Speed limit (100km/h)      							|

