import numpy as np

class NeuralNetwork:
    def _init_(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def predict(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return a2

    def train(self, X, y):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.sigmoid(z2)
        delta2 = (a2 - y) * self.derivative_sigmoid(a2)
        delta1 = (delta2.dot(self.W2.T)) * self.derivative_sigmoid(a1)
        self.W2 -= self.learning_rate * a1.T.dot(delta2)
        self.b2 -= self.learning_rate * delta2.sum(axis=0, keepdims=True)
        self.W1 -= self.learning_rate * X.T.dot(delta1)
        self.b1 -= self.learning_rate * delta1.sum(axis=0, keepdims=True)

    def cost(self, X, y):
        a2 = self.predict(X)
        return np.mean(0.5 * (a2 - y) ** 2)

def main():
    input_size = int(input("Enter the number of input features: "))
    hidden_size = int(input("Enter the number of hidden neurons: "))
    output_size = int(input("Enter the number of output neurons: "))
    learning_rate = float(input("Enter the learning rate: "))
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    num_training_examples = int(input("Enter the number of training examples: "))
    X_train = np.zeros((num_training_examples, input_size))
    y_train = np.zeros((num_training_examples, output_size))
    
    for i in range(num_training_examples):
        print(f"Enter training example {i+1} (comma-separated values for each feature): ")
        X_train[i] = np.array([float(x) for x in input().split(",")])
        print(f"Enter training label {i+1} (comma-separated values for each output): ")
        y_train[i] = np.array([float(y) for y in input().split(",")])

    num_epochs = int(input("Enter the number of epochs: "))
    for epoch in range(num_epochs):
        nn.train(X_train, y_train)
        if (epoch + 1) % 100 == 0:
            cost = nn.cost(X_train, y_train)
            print(f"Epoch {epoch + 1}, cost: {cost}")

    num_test_examples = int(input("Enter the number of test examples: "))
    X_test = np.zeros((num_test_examples, input_size))
    
    for i in range(num_test_examples):
        print(f"Enter test example {i+1} (comma-separated values for each feature): ")
        X_test[i] = np.array([float(x) for x in input().split(",")])

    predictions = nn.predict(X_test)
    print("Predictions:")
    print(predictions)

if _name_ == "_main_":
    main()
