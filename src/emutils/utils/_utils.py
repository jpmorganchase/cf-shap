"""
    Author: Emanuele Albini

    Miscellaneous  Utilities.

"""

# Python Standard Library
import os
import sys
import subprocess
import threading
from functools import reduce
import operator
from typing import Union, Iterable
import re
import datetime
from collections import defaultdict

# Basics
import pandas as pd

# NOTE: Heavy/Optional Imports are done at function-level

__all__ = [
    'prod',
    'eprint',
    'shell_command',
    'get_conda_env',
    'pandas_max',
    'remove_all_empty_spaces',
    'dict_diff',
    'timestamp',
    'show_scipy_config',
    'show_torch_config',
]

ALL_EMPTY_SPACES = re.compile(r'\s+', flags=re.UNICODE)


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H %M %S')


def prod(iterable):
    """
        Chain the * operator on a list of elements
        -> i.e. equivalent of 'sum' for + operator
    """
    return reduce(operator.mul, iterable, 1)


def shell_command(
    commands: Union[str, Iterable[str]],
    print_output: bool = True,
    timeout: Union[int, None] = None,
    kill_timeout: Union[int, None] = 10,
    verbose=2,
    **kwargs,
) -> int:
    """Run a command on the shell

    Args:
        commands (Union[str, Iterable[str]]): List of commands (List[str]) or single (str) command.
        timeout (Union[int, None], optional): Timeout (seconds) after which to kill the process. Default to None.
        kill_timeout (Union[int, None], optional): Timeout (seconds) after which to terminate the process (after kill). Default to 10 (seconds).
        print_output (bool, optional): Live-print the ouput of the commands. Defaults to True.

    Returns:
        int : Return code
    """
    if isinstance(commands, str):
        commands = [commands]

    command = ";".join(commands)

    with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            bufsize=0,
            **kwargs,
    ) as p:
        try:
            if print_output:
                # Print input commands
                for command in commands:
                    print("$ ", command)

                # Function to print the ouput (it's blocking)
                def print_output(out):
                    for c in iter(lambda: p.stdout.read(1), b''):
                        sys.stdout.buffer.write(c)
                        sys.stdout.flush()
                    out.close()

                # Start a separate thread to print the output
                print_thread = threading.Thread(target=print_output, args=(p.stdout, ))
                print_thread.start()

            # Wait for the process to end or the timeout to expire
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if verbose >= 2:
                print('Timeout Expired. Killing the process.')
            p.kill()
            try:
                p.wait(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                if verbose >= 1:
                    print('Killing Timeout Expired. Terminating the process.')
                p.kill()
                p.kill()
                p.kill()
                p.terminate()
                p.terminate()
                p.terminate()

        ret_code = p.wait()
        if verbose >= 1:
            print(f'Process existed with code {ret_code}')

        # Wait for the printing to finish
        if print_output:
            if verbose >= 3:
                print('Waiting for the output to end.')
            print_thread.join()

        return ret_code


def eprint(*args, **kwargs):
    """
    Prints on stderr
    """
    return print(*args, file=sys.stderr, **kwargs)


def get_conda_env() -> Union[str, None]:
    """Return current conda environment name

    Returns:
        Union[str, None]: Current conda environment name (if any), otherwise None.
    """
    if "envs" in sys.executable:
        return os.path.basename(os.path.dirname(os.path.dirname()))
    else:
        return None


def pandas_max(rows: Union[int, None] = None, columns: Union[int, None] = None):
    """Set the Detaframe's maximum number of rows and columns to show in Jupyter Notenooks

    Args:
        rows (Union[int, None], optional): Number of rows. Defaults to None.
        columns (Union[int, None], optional): Number of columns. Defaults to None.
    """
    if rows:
        pd.options.display.max_rows = rows
    if columns:
        pd.options.display.max_columns = columns


def remove_all_empty_spaces(sentence):
    return re.sub(ALL_EMPTY_SPACES, "", sentence)


def dict_diff(a: dict, b: dict, f=lambda x, y: x - y, default_dist=0) -> defaultdict:
    """
        Compute the difference between the pair-wise difference between two dictionaries
        
        a : dict (or defaultdict)
        b : dict (or defaultdict)
        
        Return a defaultdict (with default to 0) with results of (a - b)

        Aliases: dict_diff, dict_dif
        
    """
    c = defaultdict(lambda: default_dist)
    for key in set(a.keys()) | set(b.keys()):
        c[key] = f(a[key], b[key])
    return c


def show_scipy_config():
    import scipy
    return scipy.show_config()


def show_torch_config():
    import torch
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    from subprocess import call
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    for d in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(d)
        print(prop.name, '{:,.0f}MB'.format(prop.total_memory / (1024)**2), prop.multi_processor_count,
              'CUDA processors', 'v{0}.{1}'.format(prop.major, prop.minor))