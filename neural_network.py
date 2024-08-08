# In this we will implemenet a simple class that can be used for fully connected neural networks.
import matplotlib.pyplot as plt
import numpy as np


class FC_layer:
    def __init__(self, insize, outsize, act):
        # insize and outsie are the shapes for inputs and outputs. act is the activation function (string)
        self.act = act

        # We initialize wieghts using He initialization
        self.weights = np.random.randn(insize, outsize) * np.sqrt(2.0 / insize)
        # Biases are initialized to zeros
        self.biases = np.zeros((1, outsize))

        # We define hyperparameters for adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # We define two types of weights for propogation m is "momentum" v is "velocity"in a vague sense at least
        self.m_weights = np.zeros((insize, outsize))
        self.m_biases = np.zeros((1, outsize))
        self.v_weights = np.zeros((insize, outsize))
        self.v_biases = np.zeros((1, outsize))

    def forward_pass(self, x):
        # We define how we do a forward pass on the data x
        # Note that x will have the dimension (n x insize) n is the number of samples in this batch insize sit he dimension of insize

        self.x = x
        out = np.dot(self.x, self.weights) + self.biases
        # Now we apply the activation function

        if self.act == "relu":
            self.output = np.maximum(0, out)
        elif self.act == "softmax":
            # Note that we subtract the mean for increaced numerical stability
            exp_values = np.exp(out - np.max(out, axis=-1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        else:
            print("Invalid Activation Function!!!")
        return self.output

    def backward_pass(self, derivative, learning_rate, t):
        # derivative is the derivatives of the outputs (1, outsize), learning_rate is the hyperparameter for gradient descent
        if self.act == "softmax":
            # Go from the derivative pre softmax to post softmax
            for i, gradient in enumerate(derivative):
                if len(gradient.shape) == 1:
                    gradient = gradient.reshape(-1, 1)
                jacobian = np.diagflat(gradient) - np.dot(gradient, gradient.T)
                derivative[i] = np.dot(jacobian, self.output[i])
            # Calculate thederivative wrt weight and bias

        elif self.act == "relu":
            derivative = derivative * (self.output > 0)

        d_w = np.dot(self.x.T, derivative)
        d_b = np.sum(derivative, axis=0, keepdims=True)

        # We clip them at 1
        d_w = np.clip(d_w, -1, 1)
        d_b = np.clip(d_b, -1, 1)

        # Gradient with respect to the input
        d_i = np.dot(derivative, self.weights.T)

        # Update the weights and biases normally
        self.weights -= learning_rate * d_w
        self.biases -= learning_rate * d_b

        # We also update the weights and biases using m and v values
        m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_w
        v_weights = self.beta1 * self.v_weights + (1 - self.beta2) * (d_w**2)
        m_hat_weights = m_weights / (1 - self.beta1**t)
        v_hat_weights = v_weights / (1 - self.beta2**t)
        self.weights -= (
            learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        )

        # Finally we update the biases as well
        m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_b
        v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_b**2)
        m_hat_biases = m_biases / (1 - self.beta1**t)
        v_hat_biases = v_biases / (1 - self.beta2**t)
        self.biases -= (
            learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
        )
        # We return the derivatives wrt input for further use
        return d_i


# Finally we implement a neural network


class Fully_Connected_Network:
    def __init__(
        self, input_size, output_size, hidden_size, hidden_count, verbose=False
    ):
        # We let the final layer have softmax activation, others have relu
        # hidden size is an array
        self.layers = []
        self.verbose = verbose
        self.layers.append(FC_layer(input_size, hidden_size[0], "relu"))
        for i in range(1, hidden_count):
            self.layers.append(FC_layer(hidden_size[i - 1], hidden_size[i], "relu"))
        self.layers.append(FC_layer(hidden_size[-1], output_size, "softmax"))

    def forward_pass(self, inputs):
        # X is the data in appropriate dimensions
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def train(
        self,
        inputs,
        targets,
        n_epochs,
        initial_learning_rate,
        decay,
        plot_results=False,
    ):
        t = 0
        loss_log = []
        accuracy_log = []

        for epoch in range(n_epochs):
            # We do a forward pass
            outputs = self.forward_pass(inputs)

            # The loss we use here is categorical self entropy
            epsilon = 1e-10  # (to prevent zero logarithms)
            loss = -np.mean(targets * np.log(outputs + epsilon))

            # Calculate the accuracy
            pred_labels = np.argmax(outputs, axis=1)
            true_labels = np.argmax(targets, axis=1)
            accuracy = np.mean(pred_labels == true_labels)

            # Now we do backward propogation finally
            t += 1
            learning_rate = initial_learning_rate / (1 + decay * epoch)
            output_grad = 6 * (outputs - targets) / outputs.shape[0]
            # Now we do backward propogation in earnest
            for layer in reversed(self.layers):
                output_grad = layer.backward_pass(output_grad, learning_rate, t)

            # plot the results if needed
            if plot_results:
                loss_log.append(loss)
                accuracy_log.append(accuracy)

            # print for the current epoch
            if self.verbose:
                print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")

        if plot_results:
            plt.plot(loss_log, label="Loss")
            plt.plot(accuracy_log, label="Accuracy")
            plt.legend()
            plt.show()
        return loss_log, accuracy_log
