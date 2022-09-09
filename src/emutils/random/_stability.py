"""
    Author: Emanuele Albini

    Random stability (i.e., deterministic execution) utilities. 
"""

import logging

__all__ = [
    'random_stability',
]


def random_stability(
    seed_value=0,
    deterministic=True,
    verbose=True,
):
    '''Set random seed/global random states to the specified value for a series of libraries:

            - Python environment
            - Python random package
            - NumPy/Scipy
            - Tensorflow
            - Keras
            - Torch

        seed_value (int): random seed
        deterministic (bool) : negatively effect performance making (parallel) operations deterministic. Default to True.
        verbose (bool): Output verbose log. Default to True.
    '''
    #pylint: disable=bare-except

    outputs = []
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        outputs.append('PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        outputs.append('random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        outputs.append('NumPy')
    except:
        pass

    # TensorFlow 2
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
        if deterministic:
            outputs.append('TensorFlow 2 (deterministic)')
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.intra_op_parallelism_threads(1)
        else:
            outputs.append('TensorFlow 2 (parallel, non-deterministic)')
    except:
        pass

    # TensorFlow 1 & Keras ? Not sure it works
    try:
        import tensorflow as tf
        if tf.__version__ < '2':
            try:
                import tensorflow.compat.v1
                from tf.compat.v1 import set_random_seed
            except:
                from tf import set_random_seed

            try:
                import tensorflow.compat.v1
                from tf.compat.v1 import ConfigProto
            except:
                from tf import ConfigProto

            set_random_seed(seed_value)
            if deterministic:
                outputs.append('TensorFlow 1 (deterministic)')
                session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            else:
                outputs.append('TensorFlow 1 (parallel, non-deterministic)')
                session_conf = ConfigProto()
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            try:
                from keras import backend as K
                K.set_session(sess)
                outputs.append('Keras')
            except:
                'Keras random stability failed.'
    except:
        pass

    try:
        import torch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            outputs.append('PyTorch (deterministic)')
        else:
            outputs.append('PyTorch (parallel, non-deterministic)')

    except:
        pass
    #pylint: enable=bare-except
    if verbose:
        logging.info('Random seed (%d) set for: %s', seed_value, ", ".join(outputs))
