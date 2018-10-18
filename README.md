# TensorFlow_Python

Training first neural network using tensor flow

Source: https://www.tensorflow.org/tutorials/keras/basic_classification

This is just to understand the way tensorflow works with
 python. I would love to use tensorflow on a few Kaggle 
 Competitions.

We will be using the Fashion MNIST dataset
It contains 70,000 grayscale images in 10 categories 
We will be using 60,000 images for training and 10,000 images to evaluate
 how accurately the neural network learned to classify images.
  
 - If you get errors during the database import, 
  try using /Applications/Python\ 3.6/Install\ Certificates.command

Class names are not included in the dataset, so we store them
 
 - class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

len() and __x__.shape can be used to explore the format of the dataset.

You can easily see that the training data is of about 60,000 images.

We then preprocess the data before training the model
Using plt.show(), we can easily see that pixel value is between 0 - 255, we then scale these values to 0 to 1.

We then make sure the correct data is being trained.

We then start with the building of neural network (layers), we will be using 3 layers, flatten and two dense layers.

Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

 - Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
 - Optimizer —This is how the model is updated based on the data it sees and its loss function.
 - Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
 
Eveyrthing is ready, now we train the model.

Now we just use the model to predict the testing data and we can easily measure the accuracy.

If you print the prediction you will get a array of 10 different numbers, they describe the confidence of the image to correspond to a class

We can plot graphs to look at the 10 labels and their prediction as well to have a better understanding