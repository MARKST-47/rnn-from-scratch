import numpy as np
import matplotlib.pyplot as plt


class Tanh:
    def forward(self, inputs):
        # Apply tanh activation to the input
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # Derivative of tanh is 1 - tanh^2
        self.dinputs = dvalues * (1 - self.output**2)


class Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output > 0)


class SimpleRNN:
    def __init__(self, n_inputs, n_neurons, n_outputs, activation=Tanh()):
        # Initialize weights for input, hidden, and output layers
        self.Wx = (
            np.random.randn(n_neurons, n_inputs) * 0.1
        )  # Weights for input-to-hidden
        self.Wh = (
            np.random.randn(n_neurons, n_neurons) * 0.1
        )  # Weights for hidden-to-hidden
        self.Wy = (
            np.random.randn(n_outputs, n_neurons) * 0.1
        )  # Weights for hidden-to-output
        self.bh = np.zeros((n_neurons, 1))  # Biases for hidden layer
        self.by = np.zeros((n_outputs, 1))  # Biases for output layer
        self.activation = activation  # Use Tanh as the activation function

    def forward(self, inputs):
        T, n_features = inputs.shape
        n_neurons, _ = self.Wx.shape

        # Initialize hidden state
        self.h = np.zeros((T, n_neurons))

        # Output array
        self.y = np.zeros((T, self.Wy.shape[0]))

        # Forward pass through time
        for t in range(T):
            xt = inputs[t].reshape(-1, 1)  # (n_inputs, 1) column vector
            ht_prev = (
                self.h[t - 1].reshape(-1, 1) if t > 0 else np.zeros((n_neurons, 1))
            )  # Previous hidden state

            # Calculate hidden state using input and previous hidden state
            ht = np.dot(self.Wx, xt) + np.dot(self.Wh, ht_prev) + self.bh
            self.activation.forward(ht)
            self.h[t] = self.activation.output.flatten()

            # Calculate the output using the hidden state
            yt = np.dot(self.Wy, self.h[t].reshape(-1, 1)) + self.by
            self.y[t] = yt.flatten()

        return self.y

    def backward(self, dvalues, inputs):
        # Initialize gradients
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        T, _ = dvalues.shape
        dh_next = np.zeros((self.Wh.shape[0], 1))

        # Backpropagation through time
        for t in reversed(range(T)):
            dy = dvalues[t].reshape(-1, 1)
            dWy += np.dot(dy, self.h[t].reshape(1, -1))
            dby += dy

            dh = np.dot(self.Wy.T, dy) + dh_next
            self.activation.backward(dh)
            dht = self.activation.dinputs

            dWx += np.dot(dht, inputs[t].reshape(1, -1))
            dWh += np.dot(
                dht,
                self.h[t - 1].reshape(1, -1)
                if t > 0
                else np.zeros((dht.shape[0], 1)).T,
            )
            dbh += dht

            dh_next = np.dot(self.Wh.T, dht)

        return dWx, dWh, dWy, dbh, dby


# Stochastic Gradient Descent Optimizer
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer, dWx, dWh, dWy, dbh, dby):
        # Update the weights and biases
        layer.Wx -= self.learning_rate * dWx
        layer.Wh -= self.learning_rate * dWh
        layer.Wy -= self.learning_rate * dWy
        layer.bh -= self.learning_rate * dbh
        layer.by -= self.learning_rate * dby


def train_rnn(X_train, Y_train, rnn, optimizer, epochs=1000):
    loss_history = []
    for epoch in range(epochs):
        # Forward pass
        outputs = rnn.forward(X_train)

        # Calculate the loss (Mean Squared Error)
        loss = np.mean((outputs - Y_train) ** 2)
        loss_history.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss}")

        # Backward pass
        dvalues = 2 * (outputs - Y_train) / Y_train.size
        dWx, dWh, dWy, dbh, dby = rnn.backward(dvalues, X_train)

        # Update the parameters
        optimizer.update_params(rnn, dWx, dWh, dWy, dbh, dby)

    return loss_history
