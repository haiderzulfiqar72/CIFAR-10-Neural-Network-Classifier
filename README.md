# CIFAR-10-Neural-Network-Classifier
The CIFAR-10 Neural Network Classifier project utilizes the CIFAR-10 dataset, a widely used benchmark dataset in computer vision and machine learning. The goal is to build and train neural networks to recognize and classify images of various objects, such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The project comprises two main components: a standard fully-connected neural network and a convolutional neural network (CNN). The standard fully-connected neural network consists of an input layer with 3,072 nodes (representing the image's flattened pixel values) and multiple hidden layers with relu activation. The output layer comprises ten nodes, each representing one class, with softmax activation to produce class probabilities. The model is trained using the Adam optimizer and categorical cross-entropy loss.

The CNN architecture consists of convolutional layers with relu activation, batch normalization, and dropout to prevent overfitting. The model then flattens the output and passes it through fully-connected hidden layers with relu activation before the final output layer.

The training process involves selecting a suitable learning rate and number of epochs to optimize the model's performance. The project also employs one-hot encoding to convert class numbers to one-hot vectors, as it is preferred in neural network classification.

The performance of both models is evaluated using the CIFAR-10 test data. The classification accuracy and loss are reported, allowing for a comparison between the neural network models and previously implemented classifiers, such as the 1-NN and Bayes classifiers.

Overall, the CIFAR-10 Neural Network Classifier project provides a comprehensive exploration of neural network-based image classification on the CIFAR-10 dataset. It serves as a valuable resource for understanding and applying neural network techniques in image recognition tasks.
