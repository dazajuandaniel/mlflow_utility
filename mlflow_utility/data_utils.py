import os
import pickle
from datetime import datetime

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

