def __del__(self):
    if mlflow.active_run is not None:
        mlflow.end_run(status='KILLED')
