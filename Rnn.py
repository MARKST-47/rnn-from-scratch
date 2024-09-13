import numpy as np
import matplotlib.pyplot as plt

class RNN():
    def __init__(self, X_t, n_neurons):
        '''
        X_t: The input sequence, represented as a NumPy array.
        n_neurons: The number of neurons in the hidden layer of the RNN.
        '''
        self.T = max(X_t.shape) # lenght of input sequence
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1)) # Initializes a vector of zeros with shape (self.T, 1) to store the predicted outputs.
        self.n_neurons = n_neurons
        
        self.Wx = 0.1 * np.random.randn(n_neurons,1) # Initializes the weight matrix between the input and hidden layers using random normal values multiplied by 0.1.
        self.Wh = 0.1 * np.random.randn(n_neurons, n_neurons) # Initializes the weight matrix between the hidden layers using random normal values multiplied by 0.1.
        self.Wy = 0.1 * np.random.randn(1, n_neurons) # Initializes the weight matrix between the hidden layer and the output using random normal values multiplied by 0.1.
        self.biases = 0.1 * np.random.randn(n_neurons, 1)
        self.H = [np.zeros((self.n_neurons,1)) for t in range(self.T+1)]
        # This line initializes the hidden state of the RNN. It creates a list of self.T+1 vectors, 
        # each with shape (self.n_neurons, 1), filled with zeros. This list will store the hidden states for each time step, including an extra initial state.

    def forward(self, xt, ht_1):
        '''
        xt: The input at the current time step.
        ht_1: The hidden state from the previous time step.
        '''
        out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht_1) + self.biases
        ht = np.tanh(out)
        y_hat_t = np.dot(self.Wy, ht)
        #out is calculated by multiplying xt with self.Wx, ht_1 with self.Wh, adding the biases, and summing the results.
        #ht is obtained by applying the hyperbolic tangent activation function (np.tanh) to out.
        #y_hat_t is calculated by multiplying ht with self.Wy, which represents the predicted output at the current time step.
        return ht, y_hat_t, out
    

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

plt.plot(X_t, Y_t)
plt.show()

n_neurons = 500

rnn = RNN(X_t, n_neurons)

Y_hat = rnn.Y_hat
H = rnn.H
T = rnn.T

ht = H[0]

for t, xt in enumerate(X_t):
    xt = xt.reshape(1, 1)
    [ht, y_hat_t, out] = rnn.forward(xt, ht)
    H[t+1]   = ht
    Y_hat[t] = y_hat_t


dY = Y_hat - Y_t
L = 0.5*np.dot(dY.T,dY)/T

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()

for h in H:
    plt.plot(np.arange(20), h[0:20], 'k-', linewidth = 1, alpha = 0.05)


