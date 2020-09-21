import os
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.metrics import cohen_kappa_score,roc_auc_score,log_loss

FOLDER_DEFAULT_PATH = 'run_custom_artefacts/'
def custom_artefact_folder(run_id , type = 1):
    """
    Support function to create artefacts folders to use with Mlflow
    """

    looker = {
        1:"log_data",
        2:"log_object"
    }
    folder_path = run_id
    full_dir = FOLDER_DEFAULT_PATH+"{}/{}/v0/".format(folder_path,looker[type])
    # Try to create folder
    try:
        os.makedirs(full_dir)
    except OSError as e:
        pass
    except Exception as e:
        print(e)
        raise Exception(e)

    return full_dir
    
def serialize_for_logging(object_to_serialize, folder, name):
    """
    Function that serializes an object to use the log artifact
    Args:
        object_to_serialize: The desired object to serialize

    Returns:
        URI Path 
    """
    now = datetime.now()
    file_name = folder+name+"_"+now.strftime("%d%m%Y%H%M%S%d")+'.pkl'
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(object_to_serialize, f)
            return file_name
    except Exception as e:
        print(e)
        raise Exception("cannot serialize the object")

def _calculate_confusion_matrix(y_true, y_pred_pos, threshold = 0.5):
    """
    """
    y_pred_class = y_pred_pos > threshold
    cm = confusion_matrix(y_true, y_pred_class)
    return cm.ravel()

def score_results(y_true, y_pred_pos, threshold = 0.5, dict_res = 'confusion_matrix'):
    """
    """
    

    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = _calculate_confusion_matrix(y_true, y_pred_pos, threshold = threshold)
    
    result = {
        'confusion_matrix': (tn, fp, fn, tp),
        'fpr':fp / (fp + tn),
        'fnr':fn / (tp + fn),
        'tnr':tn / (tn + fp),
        'npv':tn/ (tn + fn),
        'fdr':fp/ (tp + fp),
        'tpr':tp / (tp + fn),
        'ppv':tp/ (tp + fp),
        'accuracy':(tp + tn) / (tp + fp + fn + tn),
        'f1score':fbeta_score(y_true, y_pred_class, beta = 1),
        'f2score':fbeta_score(y_true, y_pred_class, beta = 2),
        'cks':cohen_kappa_score(y_true, y_pred_class),
        'roc_auc': roc_auc_score(y_true, y_pred_pos),
        'log_loss':log_loss(y_true, y_pred_pos)
    }
    return result[dict_res]


