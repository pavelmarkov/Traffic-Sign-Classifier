# **Traffic Sign Recognition**

<img src="./images_for_report/traffic_signs.png" width="650" />

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

Here is a link to my [project code](https://github.com/pavelmarkov/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

- The size of training set is 34799
- The size of the validation set is 4410
- The size of test set is 12630
- The shape of a traffic sign image is (32, 32, 3)
- The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed across traffic sign labels. Where label is an integer number from 0 to 42.

<img src="./images_for_report/bar.png" width="500" />

### Design and Test a Model Architecture

#### 1. Generating additional data.

I decided to generate additional data because training data is not evenly distributed through each label.

To add more data to the data set, I just duplicated images which is underrepresented in data. So, the new image distribution became look like this:

<img src="./images_for_report/bar_new.png" width="500" />

#### 2. Data preprocessing.

As a first step, I decided to convert the images to grayscale because colour doesn't play a role in asigning a label to the traffic sign.
The second step is normalizing each image, so that the data has mean zero and equal variance: (pixel - 128)/ 128.

#### 3. Final model architecture.

My final model consisted of the following layers:

|              Layer              |             Description             |
| :-----------------------------: | :---------------------------------: |
|              Input              |    32x32x1 normalized gray image    |
| Layer 1: Convolutional 5x5x1x6  |   inputs 32x32x1, outputs 28x28x6   |
|              RELU               |           RELU Activation           |
|             Pooling             |  Input = 28x28x6. Output = 14x14x6  |
| Layer 2: Convolutional 5x5x6x16 | Input = 14x14x6. Output = 10x10x16. |
|              RELU               |           RELU Activation           |
|             Pooling             |  Input = 10x10x16. Output = 5x5x16  |
|             Flatten             |    Input = 5x5x16. Output = 400.    |
|    Layer 3: Fully Connected     |     Input = 400. Output = 120.      |
|              RELU               |           RELU Activation           |
|    Layer 4: Fully Connected     |      Input = 120. Output = 84       |
|              RELU               |           RELU Activation           |
|    Layer 5: Fully Connected     |       Input = 84. Output = 10       |

#### 3. Training neural network.

To train the model, I used following hyperparameters:

|  Hyperparameter   |     Value     |
| :---------------: | :-----------: |
| type of optimizer | AdamOptimizer |
|    batch size     |      256      |
| number of epochs  |      10       |
|   learning rate   |     0.005     |

#### 4. Finding a solution and getting the validation set accuracy to be at least 0.93.

After tuning hyperparameters (batch size, number of epochs, earning rate) and preprocessing the data I've got the following final model results:

<!-- - training set accuracy of ? -->

- validation set accuracy of 0.940
- test set accuracy of 0.909

### Test a Model on New Images

#### 1. Test the model on six German traffic signs found on the web.

Here are six German traffic signs that I found on the web:

<img src="./images_for_report/test_images.png" width="650" />

The second image (70 km/h) might be difficult to classify because it's might be difficult for the model to distinguish it from "speed limit 20 km/h" and "30 km/h" traffic sign images.

#### 2. Model's predictions on these new traffic signs.

Here are the results of the prediction:

<img src="./images_for_report/prediction.png" width="650" />

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. Whereas on the test set the accuracy is about 90%, which is slightly better.

#### 3. Softmax probabilities for each prediction. The top 5 softmax probabilities for each image.

The code for making predictions on my final model is located in the 42nd cell of the Ipython notebook.

The model is absolutely sure about all images, except for image two (for which it make a mistake) and image three (for which is the predicted label is right). As shown below in the "Top 5 labels for each image" cell, the second predicted label for the second image was 'Speed limit (70km/h)', which is its true label. The top five soft max probabilities were:

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    1.000    |               Road work               |
|    0.998    |          Go straight or left          |
|    0.993    |              Keep right               |
|    1.000    |                 Yield                 |
|    1.000    | Right-of-way at the next intersection |
|    1.000    |             Priority road             |

```
Top 5 Softmax Probabilities For Each Image:
array([[1.000, 0.000, 0.000, 0.000, 0.000],
       [0.998, 0.002, 0.000, 0.000, 0.000],
       [0.993, 0.005, 0.001, 0.001, 0.000],
       [1.000, 0.000, 0.000, 0.000, 0.000],
       [1.000, 0.000, 0.000, 0.000, 0.000],
       [1.000, 0.000, 0.000, 0.000, 0.000]], dtype=float32)
```

```
Top 5 labels for each image:
[   [   'Road work',
        'Wild animals crossing',
        'Turn right ahead',
        'General caution',
        'Bicycles crossing'],
    [   'Go straight or left',
        'Speed limit (70km/h)',
        'Turn right ahead',
        'Speed limit (30km/h)',
        'Speed limit (20km/h)'],
    [   'Keep right',
        'Speed limit (50km/h)',
        'Speed limit (80km/h)',
        'Speed limit (30km/h)',
        'Speed limit (60km/h)'],
    [   'Yield',
        'Priority road',
        'Right-of-way at the next intersection',
        'Speed limit (100km/h)',
        'Double curve'],
    [   'Right-of-way at the next intersection',
        'Double curve',
        'Priority road',
        'Beware of ice/snow',
        'Pedestrians'],
    [   'Priority road',
        'No vehicles',
        'Ahead only',
        'Roundabout mandatory',
        'Speed limit (60km/h)']]
```
