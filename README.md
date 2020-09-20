# Mlflow Utility
A wrapper function for Mlflow and Metaflow

## Installation
The package assumes you have Anaconda installed, the requirements.txt file only has the packages that are required on top of Anaconda.

## Roadmap of Elements

### Experiment
1. Run an experiment from a Script

### Data Logger
1. Support for different type of data types other than Dataframes
2. Add AutoViz Capability
3. Add capability to run data profiling workload remotely to avoid blocking current process (Async)

### Jupter Viewer
1. Add widget capability to have a proxy for mlflow UI

### Other Functionaliy
* Hyperparameter tunning using [Ray](https://docs.ray.io/en/latest/index.html)
* AutoML using