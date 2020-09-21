from abc import ABC
import pickle
from datetime import datetime

import pandas as pd
import numpy as np 

import mlflow
from  mlflow.tracking import MlflowClient
import subprocess

from . import run

def get_run_context(experiment_name):
    """
    Function that gets the context for the current experiment
    It creates a new run.
    """
    exp = Experiment(experiment_name = experiment_name)
    run = exp.start_logging(run_name = 'get_run_context')
    return run

class Experiment():
    """
    A wrapper class for MLFLOW to remove friction
    """

    def __init__(self, experiment_name = None):
        """
        Initialization Method:
        
        This method creates a new experiment or retrieve an experiment if one is running

        Args:
            experiment_name (str): The desired name for the experiment
                                   If None is provided, it will use the default Experiment

        """
        # Setup Internal variables
        self.name = 'Default' if experiment_name is None else experiment_name

        # Create a new experiment if one does't exist
        # Get list of experiments
        mlflow.set_experiment(self.name)
        self.__mlflow = mlflow
    

    def start_logging(self, run_name = None,nested = False):
        """
        Function that indicates to start a logger activity
        Args:

        Returns:
            Run Object
        """
        run_obj = run.Run(mlflow = self.__mlflow, experiment_id = self.get_experiment_id)
        run_obj.start_run(run_name = run_name, nested = nested)
        return run_obj


    def get_latest_run_id(self):
        """
        Function that returns the latest run_id
        Args:
            None
        
        Returns
            run_id (str): UUID for the last run executed 
        """
        client = self.get_client()
        runs = client.search_runs(self.get_experiment_id)[0]
        run_id = runs.to_dictionary()['info']['run_id']
        return run_id


    def get_client(self):
        """
        Function that gets the MLFLOW Client
        """
        return MlflowClient()


    def get_current_experiment_run_history(self):
        """
        Function that returns all the runs for the experiment
        Args
        """
        return self.__mlflow.search_runs()

    @property
    def get_list_of_experiments(self):
        """
        Function that gets the list of current available experiments

        Args:
            None
        
        Returns:
            exp_dict(dict): Returns a dictionary where:
                            Key: Experiment Name
                            Value: Experiment ID
        """
        return {i.name:i.experiment_id for i in MlflowClient().list_experiments()} 

    @property
    def get_experiment_id(self):
        """
        Function that gets the list of current available experiments

        Args:
            None
        
        Returns:
            exp_dict(dict): Returns a dictionary where:
                            Key: Experiment Name
                            Value: Experiment ID
        """
        self.exp_id = self.get_list_of_experiments[self.name]
        return self.exp_id


    def submit_run(self, file, parameters = None):
        """
        Function that runs a script
        Args:
            file (str): Name of the file to be submitted
            parameters (dict): Dictionary of required parameters

        Returns:

        """
        main_list = ['python',file]
        params_list = []
        if parameters is not None:
            for i in parameters.items():
                params_list.append(i[0])
                params_list.append(str(i[1]))
            final_list = main_list+params_list
        else:
            final_list = main_list

        proc = subprocess.Popen(final_list,stdout = subprocess.PIPE,stderr = subprocess.PIPE)
        stdout, stderr = proc.communicate()
        proc.terminate()
        if proc.returncode != 0:
            print(stderr.decode('UTF-8'))
        else:
            print(stdout.decode('UTF-8'))

    
