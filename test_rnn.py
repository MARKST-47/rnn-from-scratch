from rnn_final import *

# Example usage
np.random.seed(0)
T = 10   # Length of time sequence
n_inputs = 1  # Number of features
n_neurons = 5  # Number of neurons in hidden layer
n_outputs = 1  # Number of outputs

# Generate some random data for X (input) and Y (target)
X_train = np.sin(np.linspace(0, 2 * np.pi, T)).reshape(-1, 1)  # Input sequence (sin wave)
Y_train = np.cos(np.linspace(0, 2 * np.pi, T)).reshape(-1, 1)  # Target sequence (cos wave)

# Initialize the RNN and optimizer
rnn = SimpleRNN(n_inputs, n_neurons, n_outputs)
optimizer = SGD(learning_rate=0.01)

# Train the RNN
loss_history = train_rnn(X_train, Y_train, rnn, optimizer)


plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# After training, generate predictions and plot the results
def plot_predictions(rnn, X_train, Y_train):
    # Generate predictions
    predictions = rnn.forward(X_train)

    # Plot actual vs predicted values
    plt.plot(Y_train, label="Actual", linestyle='dashed')
    plt.plot(predictions, label="Predicted", linestyle='solid')
    plt.title("Actual vs Predicted Values")  
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

# Call this function after training
plot_predictions(rnn, X_train, Y_train)