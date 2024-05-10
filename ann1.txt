import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Create x values
x = np.linspace(-10, 10, 100)    #[]
print(x)

# Create plots for each activation function 
fig, axs = plt.subplots(2, 2, figsize=(8, 8)) 

axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid') 
#axs[0, 0].set(xlabel = 'x', ylabel = 'y')

axs[0, 1].plot(x, relu(x))
axs[0, 1].set_title('ReLU')

axs[1, 0].plot(x, tanh(x))
axs[1, 0].set_title('Tanh')

axs[1, 1].plot(x, softmax(x)) 
axs[1, 1].set_title('Softmax')

# Add common axis labels and titles 
fig.suptitle('Common Activation Functions')

# Show the plot 
plt.show()