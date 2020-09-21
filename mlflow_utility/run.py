from abc import ABC
import pickle
import os
from urllib.parse import unquote, urlparse



import pandas as pd
import numpy as np 
from pandas_profiling import ProfileReport

import mlflow
from  mlflow.tracking import MlflowClient


from . import data_utils

class Run():
    """
    A wrapper class for MLFLOW to remove friction
    """

    def __init__(self, experiment_id, mlflow = None ):
        """
        Initialization Method:
        
        This method creates a new experiment or retrieve an experiment if one is running

        Args:
            mlflow (str): A new mlflow element

        """
        if mlflow is None:
            raise Exception("Error, please use mlflow in constructor")
        self.mlflow = mlflow
        self.experiment_id = experiment_id

    
    def start_run(self, run_name = None, nested = False):
        """
        Function to start logging and experiment
        
        Args:
            run_name (str): Name to give the run. If the name is empty
                            If None then the user na    me is given to the run
        """
        run_name = 'user_name' if run_name is None else run_name
        self.mlflow.start_run(run_name = run_name, nested = nested)
        

    def end_run(self):
        """
        Function to end the logging capability
        """  
        self.mlflow.end_run()


    def get_active_run_attributes(self):
        """
        """
        return self.get_client().get_run(self.get_active_run_id()).data.to_dictionary()


    def get_client(self):
        """
        Function that gets the MLFLOW Client
        """
        return MlflowClient()


    def get_active_run_id(self):
        """
        Function that gets the Run ID for the ACTIVE RUN
        If there is no active run, then a message is displayed.
        Args:
            None

        Returns:
            active_run_id (str): The UUID for the active run
        """
        ar_id = self.mlflow.active_run()
        if ar_id is None:
            print ("No Active Run, please run start_run method")
            raise Exception("No Active Run, please run start_run method")
        return ar_id.info.run_id


    def log_data(self,name, df, sample = .2, report = True):
        """
        Function that logs an HTML version of 20% of the dataframe
        Args:
            sample (float): The percentage of the rows to sample
                            The sample seed is always set to 42
            report (bool): Flag to indicate if a report should be produced
                           report is produced using pandas-profiling
        """
        full_dir = data_utils.custom_artefact_folder(self.get_latest_run_id() , type = 1)
        data_set = name+'.html'
        df = df.sample(frac=sample, random_state=42)
        df.to_html(full_dir+data_set)
        self.mlflow.log_artifact(full_dir+data_set)

        if report:
            self._log_dataframe_report(df = df, name = name, sample = sample)

    def _log_dataframe_report(self,df , name, create_new_version = False, sample = 1):
        """
        Funtion that logs a datafram in the current run
        """
        full_dir = data_utils.custom_artefact_folder(self.get_latest_run_id() , type = 1)
        profile_report_name = name+ '_profiling_report.html'        
        profile = ProfileReport(df, title = profile_report_name)
        profile.to_file(full_dir+profile_report_name)
        self.mlflow.log_artifact(full_dir+profile_report_name)

    def log_object(self, obj, name):
        """
        Function that logs an object by serializing it
        Args:
            name (str): name of the object that will be serialized and logged
        """
        full_dir = data_utils.custom_artefact_folder(self.get_latest_run_id() , type = 2)
        file_name = data_utils.serialize_for_logging(object_to_serialize = obj, folder = full_dir, name = name)
        self.mlflow.log_artifact(file_name)


    def get_latest_run_id(self):
        """
        Function that returns the latest run_id
        Args:
            None
        
        Returns
            run_id (str): UUID for the last run executed 
        """
        client = self.get_client()
        runs = client.search_runs(self.experiment_id)[0]
        run_id = runs.to_dictionary()['info']['run_id']
        return run_id


    def get_latest_logged_metrics(self):
        """
        Support function to get the latest logged metrics in the last run
        Args:
            None
        Returns:
            metrics (dict): Dictionary with the logged metrics
        
        """
        run_id = self.get_latest_run_id()
        return self.mlflow.get_run(run_id).data.metrics


    def get_latest_logged_parameters(self):
        """
        Support function to get the latest logged metrics in the last run
        Args:
            None
        Returns:
            metrics (dict): Dictionary with the logged metrics
        
        """
        run_id = self.get_latest_run_id()
        return self.mlflow.get_run(run_id).data.params
    

    def get_latest_logged_artefacts(self, return_path = False):
        """
        Function that returns a list of Artefacts logged in Mlflow

        Args:
            None
        Returns:
            artefacts (list): List of artefacts logged in mlflow
        """
        run_id = self.get_latest_run_id()
        path = self.mlflow.get_run(run_id).info.artifact_uri
        parsed_path = unquote(urlparse(path).path)[1:]

        if return_path:

            return {'path':parsed_path, 'list_of_artefacts':os.listdir(parsed_path)}

        return os.listdir(parsed_path)
