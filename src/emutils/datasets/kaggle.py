# KAGGLE Message
from os import makedirs

import os

__all__ = [
    'kaggle_dataset',
]

KAGGLE_LINK = {
    'lendingclub': 'wordsforthewise/lending-club',
}


def kaggle_dataset(folder):
    os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) > 0:
        return True

    else:
        # Get the dataset short name (folder name)
        dataset = os.path.basename(os.path.dirname(os.path.join(folder, '.placeholder')))
        if dataset in KAGGLE_LINK:
            dataset_link = KAGGLE_LINK[dataset]
        else:
            raise ValueError(f'No link for this Kaggle dataset ({dataset}).')

        # Print the message to download
        print(f"""
This dataset is not directly available in this library. It will be downloaded using kaggle.
Please make sure that ~/.kaggle/kaggle.json is available and correctly configured.

To make the dataset available through this library please download it using:
    kaggle datasets download --unzip -d {dataset_link} -p {folder}


- To install kaggle use `pip install kaggle`
- If you are behind a proxy you must set it using: `kaggle config set -n proxy -v http://yourproxy.com:port`
- If you wish to download the dataset to another location other than the library location please pass an appropriate `folder` argument.
        """)

        return False