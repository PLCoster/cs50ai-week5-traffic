# CS50AI - Week 5 - Traffic

## Task:

Write an AI to identify which traffic sign appears in a photograph, using a tensorflow convolutional neural network.

## Background:

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Specification:

Complete the implementation of load_data and get_model in traffic.py.

* The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
  * You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
  * Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
  * The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
  * Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
* The get_model function should return a compiled neural network model.
  * You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
  * The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
  * The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
    * different numbers of convolutional and pooling layers
    * different numbers and sizes of filters for convolutional layers
    * different pool sizes for pooling layers
    * different numbers and sizes of hidden layers
    * dropout
* In a separate file called README.md, document (in at least a paragraph or two) your experimentation process. What did you try? What worked well? What didn’t work well? What did you notice?

Ultimately, much of this project is about exploring documentation and investigating different options in cv2 and tensorflow and seeing what results you get when you try them!

## Model Experimentation Process:

To build this model, I started with very simple models (models with small 'capacity' i.e. small number of learnable parameters), and then gradually added in more layers, increasing the complexity/capacity of the model. Each model was trained against the training data, and then evaluated using the testing data, each set of data randomly selected using Scikit-learn train_test_split (test size = 40%). I could then compare the accuracy of each model on the training set and the testing set. An ideal model would have high and similar accuracy on both the training and testing data sets.

Where a model has a higher loss on the testing data than the training data, this may suggest that the model is overfitting the training data, and so not generalising well onto the test data. When overfitting is severe, a model may be highly accurate (low loss) on the training data but have very poor accuracy (high loss) on the test data. Strategies to reduce overfitting of a model include reducing the capacity (complexity) of the model, adding 'dropout' to layers of the model, or adding weight regularization (penalizing large weights) [1](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting). However, while a simple model may reduce the risk of overfitting the training data, a model with insufficient capacity may suffer from higher loss for both the training and testing data. The capacity of the model must be tweaked to get the best results without overfitting.

There are many different model parameters that can be specified and tuned, e.g.:
* Different numbers of convolutional and pooling layers (learn features, and reduce image size/complexity)
* Different numbers and sizes of filters for convolution layers (the number of kernel matrices to train on and the size of the matrices)
* Different pool sizes for pooling layers (bigger pool size will reduce image size more)
* Different numbers and sizes of hidden layers (model complexity/capacity)
* Additional parameters for the model layers such as dropout, weight regularization, activation functions.
* Other model settings such as the optimizer algorithm, loss function and metric used to monitor the training and testing steps
* etc....!

To limit some of these choices, all models used the Adam optimisation algorithm, with categorical crossentropy for the loss function. Accuracy is a suitable metric to use for all models as we want to know the percentage of labels that were correctly predicted by the model. The output layer uses the "softmax" activation function, such that the output from the network is a normalised probability distribution (i.e. the predicted probability for the given image being each type of road sign). In addition, all hidden layers and convolutional layers will use the 'reLU' activation function.

The results of the training runs carried out can be found in the results.xlsx spreadsheet.

Some of the models explored:
#### Tiny Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```
* This model is tiny and 'simple', at least in terms of the lines of code required to make it work. However, as it is directly flattening the image data and then feeding each pixel's (30 x 30 x 3 = 2700 pixels) input to the 43 output nodes, there are more than 116 thousand weights to train for this model(!). This model performs surprisingly well, getting around 88% accuracy on the test data (average of three runs). However it could likely be improved by a more complicated model (with more hidden layers), and taking advantage of feature mapping using convolution and/or pooling layers.

#### Small-X Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(16, activation="relu"),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* The small-X model has X hidden layers of 16 units. These models were run for X = 1, 2 and 3, however these models performed poorly. Accuracy for individual runs varied wildly for these models (15% - 70%).  I suspect that reducing the 2700 pixel intensity values down to only 16 values results in losing too much information from the image, and adding extra 16-unit hidden layers does not help much. Larger hidden layers are clearly required for this task.

#### Medium-X Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```

* The medium-X model has X hidden layers of 64 units. These models performed much better than the small-X models, with test accuracy of 85%, 87% and 88% for X= 1, 2 and 3, respectively. There was much less variance between individual training runs for these models, and the training and testing accuracies were similar on all runs. Adding additional layers to the models, making them deeper, only slightly improved the accuracy of the models. These models did not seem to be overfitting the data, and were perhaps underfitting it, so some models with a larger number of units per layer were tested.

#### Large-X Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* The large-X model has X hidden layers of 128 units. These modes performed quite similarly to the medium-X models, with test accuracy of 89%, 90% and 87% for X = 1, 2 and 3 respectively. As seen for the medium models, increasing the number of hidden layers did not greatly alter the accuracy of the models. Interestingly, these models show some evidence of overfitting, as the training accuracy of these models were all over 90%, in some cases the testing accuracy was as low as 82%. These models are perhaps fitting the training data very well but not generalising well to the test data. To determine how bad this overfitting problem can get, I decided to test one final larger model with many more units per layer.


#### Huge-X Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(512, activation="relu"),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* The huge-X model has X hidden layers of 512 units. An immediate observation when running these models is how much longer training took compared to the previous models, due to the massive increase in the number of weights to be optimized during training. These models had training accuracies of around 92-93% with test accuracies of 89-91%. As seen with all previous models, adding more layers to the network did not give any real improvement to the test results. In fact one individual run of huge-3 had a training result of 93% but a test result of 86%, again perhaps evidence that these very high capacity models may overfit the training data.

### Adding Convolutional Layers:

None of the models larger than the tiny model improved much upon its test accuracy of around 88%. All the above models were only using the normalised per pixel intensity data as input, with various sizes and numbers of hidden layers before the output layer. Adding some convolutional layers will hopefully allow the models to learn multi-pixel features from the images that aid in correctly labeling the test data.

Both the number of filters to be optimised and their size can be specified, In the end I tested:
* 16, 32, 64 filters to be tested.
* Small (3x3), Medium (5x5), Large(7x7) filter sizes

These were tested on some of the differently sized models specified above. Some observations from these models were:

* Adding a 16 filter, 3x3 filter size convolutional layer to the tiny model (no hidden layers) increased training and testing accuracy to 99% and 96%.
  * Increasing the number of filters to 32 or 64 only slightly increased the testing accuracy, while greatly increasing the time taken for training.
* Adding a 16-filter, 3x3 convolutional layer to the small-1 model does not help much - the small number of units in the hidden layer result in poor accuracy and high variance in results.
  * Increasing the number of filters for the small-1 appears to somewhat improve the accuracy during some training runs but the runs still suffered from high variance.
* Adding a 16-filter, 3x3 convolutional layer to the large-1 model improved the training and testing accuracy to 99% and 96%.
  * As with the tiny model, increasing the number of filters to 32 and 64 did not have a large effect on accuracy, while greatly slowing down the training process.
* For both the tiny model and the large model, Increasing the filter size from 3x3 to 5x5 and 7x7 did not have much effect on the training or testing accuracy either, nor did it result in a significant increase in training time.


### Adding Pooling Layers:

Pooling layers are used to reduce the size of the image (by 'down-sampling'). This helps to reduce the sensitivity to the specific location of features in the input images. By adding pooling layers, the presence of features are considered in patches of the feature map rather than specific pixels. Two common pooling methods are average pooling (take average input value in the pool area) and max pooling (take the maximum input value in the pool area). The size of the pooling filter can also be specified, a common one being 2x2 with a stride of 2. The result of such a pooling layer is to reduce each dimension of the image / feature map by 2, meaning the number of pixels in the image will be reduced to a quarter the original size. [2](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

Testing out pooling layers on some of the different models with convolution above gave the following results:

* Adding a 2x2 Max pooling layer to the tiny model with 16 3x3 convolution filters resulted in training and test accuracies of 99 and 97%, a slight increase than without the pooling layer.
  * Adding more convolutional filters (32, 64) to this model made no significant difference to accuracy.
  * Switching from Max Pooling to Average Pooling for resulted in slightly lower training and testing accuracies (98% and 95.5%) for the tiny model using 15 3x3 convolution filters.
* Adding a 2x2 Max Pooling layer to the large-1 model with a 16 3x3 convolution filters had training and testing accuracies of 99% and 96%, very similar to the results for the same model without the pooling layer.
  * Using the same model with 64 3x3 convolutional filters gave a small test accuracy increase to 97%, a the cost of a much longer training time.
  * Switching to the large-3 model (3 hidden layers of 128 units), or moving to the huge-1 or huge-3 models (512 units per layer), did not result in any significant accuracy improvements.

### Multiple Convolution and Pooling Layers:

Adding multiple layers of convolution and pooling, can be used such that feature maps for gradually larger features in the images can be trained.

* When adding sequential layers of convolution and pooling, better results were generally seen when the number of filters in each convolution layer was higher.
* As with the other models being tested, adding more than one layer of hidden nodes after the convolution and pooling layers did not benefit the training or testing accuracy significantly.
* Adding more than two layers of convolution and pooling did not appear to benefit the accuracy on the models tested.
* A model found to have very high accuracy on the training and testing data was the huge-1 model with 2 sequential layers of 64 3x3 convolution and 2x2 pooling. This had mean training and testing accuracy of 99.8% and 98.9% over 3 runs.

### Adding Dropout

Adding dropout can help to avoid overfitting in neural networks, by essentially removing certain nodes during training. Although not many of the models appeared to have suffered from overfitting to the training data, adding dropout to some models was explored. When adding dropout the percentage of nodes ignored in each layer during training can be specified.

Dropout was added on the hidden layers of some of the more successful models previously tested:

* It was noticeable when adding droput to the hidden layers of the models that often the training loss/accuracy would be worse than the test loss/accuracy. As expected, adding dropout causes the model to fit less well to the training data, but improves the models ability to generalise and correctly classify the test data.
  * When adding dropout to a model with 128 units in the hidden layer, the training fit and loss were generally worse the higher the dropout was (i.e. 50% was worse than 20% dropout). This was less noticeable when carried out on the huge model with 512 units in the hidden layer. A 50% dropout on the large model will only leave 64 units during each training run, which perhaps is not sufficient, while in the case of the huge model 256 units will still be available to train.

### Overall Result

One of the best models found during the testing was:

```python
model = tf.keras.models.Sequential([

    # Add 2 sequential 64 filter, 3x3 Convolutional Layers Followed by 2x2 Pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten layers
    tf.keras.layers.Flatten(),

    # Add A Dense Hidden layer with 512 units and 50% dropout
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add Dense Output layer with 43 output units
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

This model had a training and testing accuracy of 99% , and the training and testing loss were similar during trainig runs. The model appears to fit the training data well without overfitting, and generalises well to the testing data.


## Usage:

Requires Python(3) and the python package installer pip(3) to run.

First install requirements:

$pip(3) install -r requirements.txt

Download the GTSRB dataset from https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip

Run the training and testing script:

$python3 traffic.py data_directory [model_name_to_save_model.h5]

## Acknowledgements:

Data provided by [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset#Acknowledgements)















