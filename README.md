# cs50ai-week5-traffic

## Model Experimentation Process:

To build this model, I started with very simple models (models with small 'capacity' i.e. small number of learnable parameters), and then gradually added in more layers, increasing the complexity/capacity of the model. Each model was trained against the training data, and then evaluated using the testing data, each set of data randomly selected using sklean train_test_split (test size = 40%). I could then compare the accuracy of each model on the training set and the testing set. An ideal model would have high and similar accuracy on both the training and testing data sets.

Where a model has a higher accuracy on the training data than the testing data, this may suggest that the model is overfitting the training data, and so not generalising well onto the test data. When overfitting is severe, a model may be highly accurate (low loss) on the training data but have very poor accuracy (high loss) on the test data. Strategies to reduce overfitting of a model include reducing the capacity (complexity) of the model, adding 'dropout' to layers of the model, or adding weight regularization (penalizing large weights) [1](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting). However, while a simple model may reduce the risk of overfitting the training data, a model with insufficient capacity may suffer from higher loss for both the training and testing data. The capacity of the model must be tweaked to get the best results without overfitting.

There are many different model parameters that can be specified and tuned, e.g.:
* Different numbers of convolutional and pooling layers (learn features, and reduce image size/complexity)
* Different numbers and sizes of filters for convoltion layers (the number of kernal matrices to train on and the size of the matrices)
* Different pool sizes for pooling layers (bigger pool size will reduce image size more)
* Different numbers and sizes of hidden layers (model complexity/capacity)
* Additional parameters for the model layers such as dropout, weight regularization, activation functions.
* Other model settings such as the optimizer algorithm, loss function and metric used to monitor the training and testing steps
* etc....!

To limit some of these choices, all models used the Adam optimisation algorithm, with categorical crossentropy for the loss function. Accuracy is a suitable metric to use for all models as we want to know the percentage of labels that were correctly predicted by the model. The output layer uses the "softmax" activation function, such that the output from the network is a normalised probability distribution (i.e. the predicted probability for the given image being each type of road sign). In addtion, all hidden layers and convolutional layers will use the 'reLU' activation function.

Some of the models explored:
#### Tiny Model:
```python
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
      tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```
* This model is tiny and 'simple', at least in terms of the lines of code required to make it work. However, as it is directly flattening the image data and then feeding each pixel's (30 x 30 x 3 = 2700 pixels) input to the 43 output nodes, there are more than 116 thousand weights to train for this model(!). This model performs surprisingly well, getting around 88% accurracy on the test data (average of three runs). However it couls likely be improved by a more complicated model (with more hidden layers), and taking advantage of feature mapping using convolution and/or pooling layers.

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

* The huge-X model has X hidden layers of 512 units. An immediate observation when running these models is how much longer training took compared to the previous models, due to the massive increase in the number of weights to be optimized during training. These models had training accuracies of around 92-93% with test accuracies of 89-91%. As seen with all pervious models, adding more layers to the network did not give any real improvement to the test results. In fact one individual run of huge-3 had a training result of 93% but a test result of 86%, again perhaps evidence that these very high capacity models may overfit the training data.

### Adding Convolutional Layers:

None of the models larger than the tiny model improved much upon its test accuracy of around 88%. ALl the above models were only using the normalised per pixel intensity data as input, with various sizes of numbers of hidden layers before the output layer. Adding some convolutional layers will hopefully allow the models to learn multi-pixel features from the images that aid in correctly labelling the test data. Pooling layers can also be added to reduce the size of the input data to the hidden layers of the model, which may help to speed up the training times.

Both the number of filters to be optimised and their size can be specified, In the end I tested:
* 16, 32, 64 filters to be tested.
* Small (3x3), Medium (5x5), Large(7x7) filter sizes

These were tested on some of the differently sized models specified above. Some observations from these models were:

* Adding a 16 filter, 3x3 filter size convolutional layer to the tiny model (no hidden layers) increased training and testing accuracy to 99% and 96%.
  * Increasing the number of filters only slightly increased the testing accuracy, while greatly increasing the time taken for training.
* Adding a 16-filter, 3x3 convolutional layer to the small-1 model does not help much - the small number of units in the hidden layer result in poor accuracy and high variance in results.
  * Increasing the number of filters for the small-1 appears to somewhat improve the accuracy during some training runs but the runs still suffered from high variance.
* Adding a 16-filter, 3x3 convolutional layer to the large-1 model improved the training and testing accuracy to 99% and 96%.
  * As with the tiny model, increasing the number of filters to 32 and 64 did not have a large effect on accuracy, while greatly slowing down the training process.
* For both the tiny model and the large model, Increasing the filter size from 3x3 to 5x5 and 7x7 did not have much effect on the training or testing accuracy either, nor did it result in a significant increase in training time.


### Adding Pooling Layers:

Pooling layers are used to reduce the size of the image (by 'down-sampling'). This helps to reduce the sensitivity to the specific location of features in the input images. By adding pooling layers, the prescence of features are considered in patches of the feature map rather than specific pixels. Two common pooling methods are average pooling (take average input value in the pool area) and max pooling (take the maximum input value in the pool area). The size of the pooling filter can also be specified, a common one being 2x2 with a stride of 2. The result of such a pooling layer is to reduce each dimension of the image / feature map by 2, meaning the number of pixels in the image will be reduced to a quarter the original size. [2](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

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

### Adding Dropout

Adding dropout can help to avoid overfitting in neural networks, by essentially removing certain nodes during training. Although not many of the models appeared to have suffered from overfitting to the training data, adding dropout to some models was explored. When adding dropout the percentage of nodes ignored in each layer during training can be specified.

Dropout was added on some of the more successful models previously tested:














