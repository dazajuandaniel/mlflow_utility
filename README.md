# Mlflow Utility
A wrapper function for Mlflow and Metaflow

## Installation
The package assumes you have Anaconda installed, the requirements.txt file only has the packages that are required on top of Anaconda.

## Roadmap of Elements

### Experiment
1. Run an experiment from a Script (`partially done`)

### Data Logger
1. Support for different type of data types other than Dataframes (i.e. numpy)
2. Add capability to run data profiling workload remotely to avoid blocking current process (Async)

### Jupter Viewer
1. Add widget capability to have a proxy for mlflow UI

### AutoML
1. Add logging for all runs in mlflow
2. Add capability to test on parameter dataset.
3. Add feature tools integration
4. Add Shap
5. Add Feature Selection

### Other Functionaliy
* Add proper logging to remove print statements
* Hyperparameter tunning using [Ray](https://docs.ray.io/en/latest/index.html)