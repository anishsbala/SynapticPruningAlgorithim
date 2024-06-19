# GitHub Codespaces ♥️ Jupyter Notebooks

Synaptic pruning is an essential part of brain development in early childhood and adulthood, where redundant or extra synapses are removed, optimizing neural circuitry and cognitive function. Drawing inspiration from this process, this research aims to apply a similar approach to creating sparse networks within dense neural networks. By mimicking synaptic pruning, unnecessary connections can systematically be eliminated, resulting in more efficient and compact artificial neural networks. This paper will explore the benefits of such pruning techniques through the MNIST dataset to decrease computational requirements and significantly reduce the model's footprint. 

### Installation

# Clone the repository
git clone https://github.com/adit-bala/chores.git

To replicate this experiment, a Python environment must be set up with the necessary packages. These can be installed using the following command: pip install tensorflow ipywidgets matplotlib seaborn statsmodels tabulate. This experiment utilizes the MNIST dataset, which is publically available through  TensorFlow's Keras API. The dataset will be automatically downloaded and preprocessed when running the code. In terms of how the data is handled:

Raw Data: The MNIST dataset is downloaded from the Keras dataset repository.

Preprocessed Data: The images are normalized to the range [0, 1] by dividing by 255.0 and reshaped to a format suitable for input to a dense neural network.

Intermediate Data: During training, model weights and performance metrics are saved and updated.

Generated Data: Final model weights after pruning and accuracy/loss metrics for each pruning iteration are generated.

To reproduce the results of the experiment from loading the data to the pruning and evaluation, copy the entirety of the code within the 'synapticprunealgo (1).py' file into the Jupyter notebook 'matplotlib.ipynb.' Altogether, this will load and preprocess the MNIST dataset, create a dense artificial neural network, train the model and save the initial weights, prune the model iteratively and evaluate performance, and plot/ display the results. 

# Chore Calendar Generator

short script to schedule chores with roommates and export data to an .isc file

# How To Use

### Installation

```
# Clone the repository
git clone https://github.com/adit-bala/chores.git

# Enter into the directory
cd chores/

# Install the dependencies
pip install ics
```

### Modifying

To personalize the calendar, adjust the lines that have the `# UPDATE` above them with the appropriate value 

### Exporting

To export your calendar to an .ics file, run `chores.py` in your terminal

```
python3 chores.py
```

Import `chores_calendar.ics` into Google Calendar.





 
