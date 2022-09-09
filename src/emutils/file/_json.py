import json
import numpy as np

__all__ = ['load_json', 'save_json']


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def save_json(obj, filename):
    with open(filename, "w") as handle:
        json.dump(obj, handle, cls=NumpyJSONEncoder, indent=4)


def load_json(filename):
    with open(filename, "r") as handle:
        return json.load(handle)