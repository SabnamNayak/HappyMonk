# HappyMonk Assignment
# Neural Network Model with Adaptive Activation (Ada-Act)
[Click Here]([https://scikit-optimize.github.io/stable/](https://github.com/SabnamNayak/HappyMonk/blob/main/activation_function.ipynb))

The code implements a 1-hidden layer neural network model that adapts its activation function based on the dataset. The activation function used is called Ada-Act (Adaptive Activation). It follows a flexible functional form of `k0 + k1 * x`, where `k0` and `k1` are parameters that are learned from multiple runs of the algorithm.

The code includes the following functions:
- `initialize_network`: Initializes the neural network with random weights.
- `activate`: Calculates the activation of a neuron using the Ada-Act activation function.
- `transfer`: Transfers the activation from a neuron to the next layer.
- `forward_propagate`: Performs forward propagation of input through the network.
- `backward_propagate_error`: Backpropagates the error through the network and updates the weights.
- `update_weights`: Updates the weights of the network based on the backpropagated error.
- `train_network`: Trains the network for a fixed number of epochs using the training dataset.
- `predict`: Makes predictions on new input data using the trained network.
- `evaluate_algorithm`: Evaluates the algorithm by running it on a dataset and calculating the accuracy of its predictions.

The main function, `back_propagation`, implements the backpropagation algorithm with the Ada-Act activation function. It takes the training dataset, test dataset, learning rate, number of epochs, number of hidden neurons, and the parameters `k0` and `k1` as input. It initializes the network, trains it using the training dataset, and makes predictions on the test dataset. The predictions are returned as the output.

The code also includes helper functions for loading and preparing the dataset, converting the class column to integers, normalizing the input variables, and calculating the mean accuracy of the algorithm's predictions.

Overall, the code aims to create a neural network model that dynamically adapts its activation function using the Ada-Act approach, allowing it to learn the most suitable activation function for a given dataset.

## Automating the Selection of Activation Function
[Click Here...]([https://scikit-optimize.github.io/stable/](https://github.com/SabnamNayak/HappyMonk/blob/main/tried_error.ipynb))

To automatically determine the best activation function without using grid search or brute force, you can consider using a meta-learning approach or an optimization algorithm. One popular approach is to use a meta-learning algorithm called Bayesian Optimization. Bayesian Optimization is an efficient and effective method for optimizing hyperparameters based on a probabilistic model of the objective function.

Here's an outline of the steps involved in using Bayesian Optimization to find the best activation function:
1. Define the search space: Specify the set of activation functions you want to consider. This can include popular choices such as sigmoid, tanh, ReLU, Leaky ReLU, and others.
2. Define the objective function: Choose a performance metric, such as accuracy or F1 score, that you want to maximize.
3. Set up the Bayesian Optimization framework: Use a library or framework that provides Bayesian Optimization functionality. One popular library is scikit-optimize (skopt) in Python.
4. Define the search space and optimization constraints: Specify the search space for the hyperparameters, including the activation function and any other relevant hyperparameters. Set any necessary constraints or bounds.
5. Define the objective function for optimization: Write a function that takes the activation function and other hyperparameters as input, trains a neural network using those hyperparameters, and returns the performance metric to be maximized.
6. Run the optimization: Use the Bayesian Optimization framework to search for the best activation function by repeatedly evaluating the objective function with different hyperparameters. The framework will suggest new hyperparameters to evaluate based on the previous results.
7. Evaluate and select the best activation function: After the optimization process is complete, evaluate the performance of the neural network with the suggested best activation function on a separate validation set. This will give you an estimate of its performance.
