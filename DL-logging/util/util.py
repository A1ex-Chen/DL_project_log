import json
import os
import polars as pl
from typing import List

from config.constant import PATH_FILE
from util.log import configure_logger

logger = configure_logger('github-data_logger', 'logging_file.log')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root


def save_to_json(data, file, mode='w') -> None:
    with open(file, mode) as filey:
        filey.writelines(json.dumps(data, indent=4))


def read_json(filepath: str) -> List[dict]:
    """
    This function reads a json file and returns its contents as a dictionary.
    It accepts one argument:
    filepath (str) : The path of the json file that needs to be read
    If the file is not found or is not a valid json file, it returns None
    """
    try:
        print(ROOT_DIR)
        print(PATH_FILE['data'] + filepath)
        with open(ROOT_DIR + '/' + PATH_FILE['data'] + filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
    except json.decoder.JSONDecodeError:
        logger.error(f"Invalid JSON file: {filepath}")


def save_to_csv(json_list: List[dict], library) -> None:
    """
    This function takes a json list and saves the path, html_url and repository full_name in a csv file.
    json_list (list) : List of json data
    """
    headers = ['library', 'path', 'html_url', 'repository_full_name']
    rows = []
    for i in library:
        for item in json_list:
            row = [i, item['path'], item['html_url'], item['repository']['full_name']]
            rows.append(row)


    df = pl.DataFrame(rows, schema=headers)
    df.write_csv(ROOT_DIR + '/' + PATH_FILE['data'] + 'repo.csv')
    logger.info("Data saved to output.csv")


def get_json_files(directory: str) -> List[str]:
    """
    This function reads a directory and returns a list of json file names in the directory
    directory (str) : The path of the directory that needs to be read
    """
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_files.append(filename)
    return json_files


def find_notebooks_recursive(folder_path: str, ext: str) -> list:
    """
    Recursively finds all Jupyter notebook files with a specified extension in a specified folder and all its subfolders.

    Args:
    - folder_path (str): The path to the folder to search.
    - ext (str): The extension of the notebook files to search for (default: '.ipynb').

    Returns:
    - A list of paths to all the Jupyter notebook files found.
    """
    notebooks = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            notebooks.extend(find_notebooks_recursive(item_path, ext))
        elif item.endswith(ext):
            notebooks.append(item_path)
    return notebooks


def save_to_csv_function_call(data: list, file_path: str) -> None:
    """
    Saves a list of tuples to a CSV file.

    Args:
    - data (list): A list of tuples to be saved to the CSV file.
    - file_path (str): The path to the CSV file to create.

    Returns:
    - None
    """

    df = pl.DataFrame(data)
    df.write_csv(file_path)


def list_subfolders(folder_path):
    subfolders = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isdir(path):
            subfolders.append(path)
    return subfolders

