def check_dataset(path: str, task: str) ->None:
    """
    Function for error-checking HUB dataset Zip file before upload. It checks a dataset for errors before it is uploaded
    to the HUB. Usage examples are given below.

    Args:
        path (str): Path to data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify', 'obb'.

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
        ```python
        from ultralytics.hub import check_dataset

        check_dataset('path/to/coco8.zip', task='detect')  # detect dataset
        check_dataset('path/to/coco8-seg.zip', task='segment')  # segment dataset
        check_dataset('path/to/coco8-pose.zip', task='pose')  # pose dataset
        check_dataset('path/to/dota8.zip', task='obb')  # OBB dataset
        check_dataset('path/to/imagenet10.zip', task='classify')  # classification dataset
        ```
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(
        f'Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.'
        )
