import os

def custom_artefact_folder(run_id , type = 1):
    """
    Support function to create artefacts folders to use with Mlflow
    """

    looker = {
        1:"log_data",
        2:"log_other"
    }
    folder_path = run_id
    full_dir = "run_custom_artefacts/{}/{}/v0/".format(folder_path,looker[type])
    # Try to create folder
    try:
        os.makedirs(full_dir)
    except OSError as e:
        pass
    except Exception as e:
        print(e)
        raise Exception(e)

    return full_dir
    

