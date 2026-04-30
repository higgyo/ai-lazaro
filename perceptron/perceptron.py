import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        # Initialize weights with random values between 0 and 1
        # We need 4 weights: w1, w2, w3 and w0 (bias)
        # Assuming input shape is [x1, x2, x3], we'll treat x0 as -1 for the bias term
        self.weights = np.random.uniform(0, 1, 4)
        
        # Keep track of weight history for plotting
        self.weights_history = []
        
    def predict(self, x):
        # x is an array [x1, x2, x3]
        # x_with_bias = [x1, x2, x3, -1]
        x_with_bias = np.append(x, -1)
        v = np.dot(self.weights, x_with_bias)
        # Activation function g(.)
        return 1 if v >= 0 else -1
        
    def fit(self, X, d):
        """
        Train the perceptron using Hebbian learning rule.
        X: array of shape (n_samples, 3)
        d: array of shape (n_samples,) representing desired classes (-1 or 1)
        """
        # Save initial weights
        self.weights_history.append(self.weights.copy())
        
        epochs = 0
        error = True
        
        while error:
            error = False
            for i in range(len(X)):
                x = X[i]
                desired_d = d[i]
                
                # Check current prediction
                y = self.predict(x)
                
                if y != desired_d:
                    error = True
                    # Hebbian rule for supervised learning: w_new = w_old + learning_rate * d * x
                    x_with_bias = np.append(x, -1)
                    self.weights = self.weights + self.learning_rate * desired_d * x_with_bias
                    
            epochs += 1
            self.weights_history.append(self.weights.copy())
            
        return epochs
