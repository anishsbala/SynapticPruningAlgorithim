### GitHub Codespaces ♥️ Jupyter Notebooks

# Abstract

Synaptic pruning is an essential part of brain development in early childhood and adulthood, where redundant or extra synapses are removed, optimizing neural circuitry and cognitive function. Drawing inspiration from this process, this research aims to apply a similar approach to creating sparse networks within dense neural networks. By mimicking synaptic pruning, unnecessary connections can systematically be eliminated, resulting in more efficient and compact artificial neural networks. This paper will explore the benefits of such pruning techniques through the MNIST dataset to decrease computational requirements and significantly reduce the model's footprint. 

# Replication

```
# Clone the repository
git clone https://github.com/anishsbala/SynapticPruningAlgorithim.git

# Enter into the directory
cd SynapticPruningAlgorithim/

#Install the necessary packages
pip install tensorflow ipywidgets matplotlib seaborn statsmodels tabulate
```

This experiment utilizes the MNIST dataset, which is publicly available through TensorFlow's Keras API. The dataset will be automatically downloaded and preprocessed when running the code. In terms of how the data is handled:

1. Raw Data: The MNIST dataset is downloaded from the Keras dataset repository.
   
2. Preprocessed Data: The images are normalized to the range [0, 1] by dividing by 255.0 and reshaped to a format suitable for input to a dense neural network.
3. Intermediate Data: During training, model weights and performance metrics are saved and updated.
4. Generated Data: Final model weights after pruning and accuracy/loss metrics for each pruning iteration are generated.

To reproduce the results of the experiment from loading the data to the pruning and evaluation, copy the entirety of the code within the 'synapticprunealgo (1).py' file into the Jupyter notebook 'matplotlib.ipynb.' Altogether, this will:

1. Load and preprocess the MNIST dataset.
   
2. Create a dense artificial neural network.
3. Train the model and save the initial weights.
4. Prune the model iteratively and evaluate performance.
5. Plot and display the results.






 
