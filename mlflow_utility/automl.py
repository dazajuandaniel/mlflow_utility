from datetime import datetime
import tpot
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix
from . import data_utils, experiment, run


METRICS = ['confusion_matrix','fpr','fnr',
           'tnr','npv','fdr','tpr','ppv',
           'accuracy','f1score','f2score','cks','roc_auc','log_loss']


class AutoML():
    """
    A wrapper class for TPOT to remove friction
    """

    def __init__(self,
                 name, 
                 training_data,
                 training_target,
                 validation_data,
                 validation_target,
                 task = 'classification',
                 iterations=100,
                 primary_metric = 'accuracy',
                 **kwargs):
        """
        Class that instatiates the Automl Class
        For full details of the available parameters, please go to TPOT's doc site.
        """
        # Setup Mlflow Logging
        exp = experiment.Experiment(experiment_name = name)
        self.exp = exp
        self.name = name
        # Load kwargs values
        generations = kwargs.get('generations',100)
        population_size = iterations
        offspring_size = kwargs.get('offspring_size',None)
        mutation_rate = kwargs.get('mutation_rate',0.9)
        crossover_rate = kwargs.get('crossover_rate',0.1)
        cv = kwargs.get('cv',5)
        max_time_mins = kwargs.get('max_time_mins',None)
        max_eval_time_mins = kwargs.get('max_eval_time_mins',5)
        config_dict = kwargs.get('config_dict',None)
        periodic_checkpoint_folder = kwargs.get('periodic_checkpoint_folder',None)
        early_stop = kwargs.get('early_stop',None)
        verbosity = kwargs.get('verbosity',0)


        self.tpot_clsf = TPOTClassifier(generations=generations,
                                        verbosity=verbosity,
                                        population_size=population_size,
                                        offspring_size=offspring_size,
                                        mutation_rate = mutation_rate,
                                        crossover_rate=crossover_rate,
                                        cv=cv,
                                        max_time_mins=max_time_mins,
                                        max_eval_time_mins=max_eval_time_mins,
                                        config_dict=config_dict,
                                        periodic_checkpoint_folder=periodic_checkpoint_folder,
                                        early_stop=early_stop,  
                                        n_jobs=-1)

        # Log Data Artefacts
        self.training = training_data
        self.training_target = training_target

        self.validation = validation_data
        self.validation_target = validation_target


    def fit(self):
        """
        """
        # Start Logging
        now = datetime.now()
        run = self.exp.start_logging(run_name = self.name+'_run_automl_'+now.strftime("%d%m%Y%H%M%S%d"))

        # Log Parameters
        self.run = run
        for i in self.tpot_clsf.get_params().items():
            self.run.mlflow.log_param(i[0],str(i[1]))
        
        # Log Data
        run.log_data(df = self.training, name = 'training_data')
        run.log_data(df = self.training_target.to_frame(), name = 'training_target')

        self.tpot_clsf.fit(self.training, self.training_target)

    def score(self,x = None, y = None):
         """
         Function that calculates the score based on TPOTs optimization function
         Args:
            x: Feature Space
            y: Target
         """
         x = self.training if x is None else x
         y = self.training_target if y is None else y
         score = (self.tpot_clsf.score(x, y)) 
         self.run.mlflow.log_metric('tpot_score',score)
         return score

    def predict(self,x = None):
        """
        Funtion that calculates the predicted probabilities.
        Uses the validation dataset by default
        
        Args:
            x: Feature Space

        Returns
            predictions (list): An array with the predicted classes
        """
        x = self.validation if x is None else x
        try:
            predictions = self.tpot_clsf.predict(x)
            return predictions
        except Exception as e:
            print(e)
            print("Please fit an AutoML object first")
    
    def predict_proba(self, x = None):
        """
        Funtion that calculates the predicted probabilities.
        Uses the validation dataset by default
        
        Args:
            x: Feature Space

        Returns
            probabilities (list): An array with the positive class probabilities
        """
        x = self.validation if x is None else x
        try:
            predictions = self.tpot_clsf.predict_proba(x)[:,1]
            return predictions
        except Exception as e:
            print(e)
            print("Please fit an AutoML object first") 
    
    def get_all_scores(self,y = None, threshold = 0.5):
        """
        Function that gets all scores at once, it is based on the following article:
        URL: https://towardsdatascience.com/the-ultimate-guide-to-binary-classification-metrics-c25c3627dd0a

        Args:
            y: True values
            threshold (float): The threshold to determine the outcome class
        
        Returns:
            None
        """
        y = self.validation_target if y is None else y
        probas = self.predict_proba()
        for i in METRICS:
            score = data_utils.score_results(y, probas, threshold = threshold, dict_res = i)
            try:
                self.run.mlflow.log_metric('tpot_score_'+i,score)
            except Exception as e:
                print(e)
                pass

            print(i,score)
    
    def export(self, file_name):
        """
        Funtion to export the code generated by TPOT
        Args:
            file_name (str): The desired name for the .py file

        Returns:
            None
        """
        self.tpot_clsf.export(file_name)
        run.mlflow.log_artifact(file_name)

