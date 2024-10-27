import numpy as np
import matplotlib.pyplot as plt
import src

input_vectors = np.array([[3, 1.5],[2, 1],[4, 1.5],[3, 4],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1]])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

#create a neural network object and make a prediction using the randomly assigned
#weights and bias parameters
nn = src.NeuralNetwork(learning_rate = 0.1)
print(nn.predict(input_vectors))

#train the model and save a graph of cumulative error
training_error = nn.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

#make new predictions based on the updated weights and bias parameters
print(nn.predict(input_vectors))