"""Module to return the location of the Kaggle dataset. Or prompt the user to download it."""

from typing import Union
import os

from .. import PACKAGE_DATA_FOLDER

__author__ = 'Emanuele Albini'
__all__ = [
    'kaggle_dataset',
]

KAGGLE_LINK = {
    'lendingclub': 'wordsforthewise/lending-club',
}


def kaggle_dataset(
    dataset: str,
    directory: Union[None, str] = None,
    base_path: str = PACKAGE_DATA_FOLDER,
) -> str:
    """Returns the location of the Kaggle dataset. Or prompt the user to download it, otherwise.

    Args:
        dataset (str): The handle/key/short name of the dataset. (e.g., 'lendingclub')
        directory (Union[None, str], optional): The directory where to download the dataset files. Defaults to None ( = dataset handle).
        base_path (str, optional): The data directory base path. Defaults to PACKAGE_DATA_FOLDER.

    Returns:
        str: The location of the dataset, if available, otherwise None.
    """

    # If the directory name is not passed we use the dataset name
    if directory is None:
        directory = dataset

    path = os.path.abspath(os.path.join(base_path, directory))

    # Create the directory where to store the dataset, if it doesn't exist
    os.makedirs(path, exist_ok=True)

    if len(os.listdir(path)) > 0:
        return path

    else:
        # Get the dataset short name (folder name)
        if dataset in KAGGLE_LINK:
            dataset_link = KAGGLE_LINK[dataset]
        else:
            raise ValueError(f'No link for this Kaggle dataset ({dataset}).')

        # Print the message to download
        print(f"""
This dataset is not readily available in this package. It must be downloaded using Kaggle API.
Please make sure that ~/.kaggle/kaggle.json is correctly configured.

To make the dataset available through this package please download it using the following command.
    kaggle datasets download --unzip -d {dataset_link} -p {path}


- To install kaggle use `pip install kaggle`
- If you are behind a proxy you must set it using: `kaggle config set -n proxy -v http://yourproxy.com:port`
- If you wish to download the dataset to another location other than the default location please pass appropriate `directory` and `base_path` arguments.
        """)

        return None