# Ensure TensorFlow and other necessary libraries are installed
%pip install tensorflow ipywidgets matplotlib seaborn statsmodels tabulate

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import defaultdict
from tabulate import tabulate  # for generating tables as text

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for compatibility with Dense layers
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Create a simple dense neural network model
def create_dense_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Compile and train the dense model
dense_model = create_dense_model()
dense_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = dense_model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test))

# Save the initial weights of the dense model
initial_weights = dense_model.get_weights()

# Helper function for smoother lines using LOWESS
def smooth_line(x, y, frac=0.1):
    # Remove NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid_indices], y[valid_indices]
    
    # Ensure x and y are not empty after cleaning
    if len(x) == 0 or len(y) == 0:
        return x, y
    
    smoothed = lowess(y, x, frac=frac)
    return smoothed[:, 0], smoothed[:, 1]

# Plot initial training results
epochs = np.arange(len(history.history['accuracy']))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
x_smooth, y_smooth = smooth_line(epochs, np.array(history.history['accuracy']))
plt.plot(x_smooth, y_smooth, label='Train Accuracy', color='blue')
x_smooth, y_smooth = smooth_line(epochs, np.array(history.history['val_accuracy']))
plt.plot(x_smooth, y_smooth, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
x_smooth, y_smooth = smooth_line(epochs, np.array(history.history['loss']))
plt.plot(x_smooth, y_smooth, label='Train Loss', color='blue')
x_smooth, y_smooth = smooth_line(epochs, np.array(history.history['val_loss']))
plt.plot(x_smooth, y_smooth, label='Test Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Loss over Epochs')

plt.suptitle('Initial Model Training')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Pruning function
def prune_weights(model, pruning_fraction):
    total_weights = 0
    total_pruned = 0
    
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            total_weights += weights.size
            k = int(pruning_fraction * weights.size)
            flat_weights = weights.flatten()
            idx = np.argsort(np.abs(flat_weights))[:k]
            total_pruned += len(idx)
            flat_weights[idx] = 0
            pruned_weights = flat_weights.reshape(weights.shape)
            layer.set_weights([pruned_weights, biases])
    
    return model, total_weights, total_pruned

# Function to create winning tickets with the Lottery Ticket Hypothesis
def create_winning_tickets(model, x_train, y_train, x_test, y_test, initial_weights, iterations=5, pruning_fractions=[0.2]):
    results = {pf: {'accuracies': [], 'losses': [], 'weights_retained': []} for pf in pruning_fractions}
    
    final_pruned_models = {}
    for pruning_fraction in pruning_fractions:
        pruned_model = create_dense_model()
        pruned_model.set_weights(initial_weights)

        for i in range(iterations):
            pruned_model, total_weights, total_pruned = prune_weights(pruned_model, pruning_fraction)
            pruned_model.set_weights(initial_weights)  # Reset to initial weights
            pruned_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            pruned_model.fit(x_train, y_train, epochs=1, verbose=0)
            loss, accuracy = pruned_model.evaluate(x_test, y_test, verbose=0)
            results[pruning_fraction]['accuracies'].append(accuracy)
            results[pruning_fraction]['losses'].append(loss)
            results[pruning_fraction]['weights_retained'].append((total_weights - total_pruned) / total_weights)
            print(f"Pruning Fraction {pruning_fraction}, Iteration {i + 1}, Accuracy: {accuracy}, Loss: {loss}")
        
        final_pruned_models[pruning_fraction] = pruned_model
    
    return results, final_pruned_models

# Perform pruning and plot results
pruning_iterations = 5
pruning_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
results, final_pruned_models = create_winning_tickets(dense_model, x_train, y_train, x_test, y_test, initial_weights, pruning_iterations, pruning_fractions)

# Helper function to handle duplicates and sort
def unique_sorted(x, y):
    unique_x = np.unique(x)
    y_averaged = np.array([np.mean(y[x == ux]) for ux in unique_x])
    return unique_x, y_averaged

# Plotting the results
fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# Plot accuracies with error bars
for pf in pruning_fractions:
    accuracies = results[pf]['accuracies']
    x_smooth, y_smooth = smooth_line(np.arange(1, pruning_iterations + 1), np.array(accuracies))
    axes[0].plot(x_smooth, y_smooth, label=f'Pruning Fraction {pf}')
    axes[0].fill_between(range(1, pruning_iterations + 1), 
                         np.array(accuracies) - np.std(accuracies), 
                         np.array(accuracies) + np.std(accuracies), alpha=0.2)

# Plot losses with error bars
for pf in pruning_fractions:
    losses = results[pf]['losses']
    x_smooth, y_smooth = smooth_line(np.arange(1, pruning_iterations + 1), np.array(losses))
    axes[1].plot(x_smooth, y_smooth, label=f'Pruning Fraction {pf}')
    axes[1].fill_between(range(1, pruning_iterations + 1), 
                         np.array(losses) - np.std(losses), 
                         np.array(losses) + np.std(losses), alpha=0.2)

# Plot weight retention vs accuracy
for pf in pruning_fractions:
    accuracies = results[pf]['accuracies']
    weights_retained = results[pf]['weights_retained']
    unique_wr, unique_acc = unique_sorted(np.array(weights_retained), np.array(accuracies))
    x_smooth, y_smooth = smooth_line(unique_wr, unique_acc)
    axes[2].plot(x_smooth, y_smooth, label=f'Pruning Fraction {pf}')
    axes[2].scatter(weights_retained, accuracies)

# Customize the plots
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='best')
axes[0].set_title('Model Accuracy vs Pruning Iterations')
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

axes[1].set_ylabel('Loss')
axes[1].legend(loc='best')
axes[1].set_title('Model Loss vs Pruning Iterations')
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

axes[2].set_xlabel('Weights Retained')
axes[2].set_ylabel('Accuracy')
axes[2].legend(loc='best')
axes[2].set_title('Model Accuracy vs Weights Retained')
axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# Display final results in tabular format
table_data = []
for pf in pruning_fractions:
    for i in range(pruning_iterations):
        table_data.append([pf, i+1, results[pf]['accuracies'][i], results[pf]['losses'][i], results[pf]['weights_retained'][i]])

headers = ['Pruning Fraction', 'Iteration', 'Accuracy', 'Loss', 'Weights Retained']
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Interactive function to visualize training samples
def visualize_training_samples(sample_indices):
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, len(sample_indices), i + 1)
        plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y_train[idx]}')
        plt.axis('off')
    plt.show()

# Interact with training samples
interact(visualize_training_samples, sample_indices=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
