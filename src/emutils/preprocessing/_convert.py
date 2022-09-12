"""
    Author: Emanuele Albini

    Basic preprocessing utilities for datasets.
"""

from typing import Union, Iterable

import numpy as np
import pandas as pd

PROCESS_TYPE = {pd.DataFrame: 'pd', np.ndarray: 'np', dict: 'dict'}

__all__ = [
    'process_data',
    'data_astype',
]


def __to_tensor(X):
    from tensorflow import convert_to_tensor
    return convert_to_tensor(X)


def data_astype(
    X: Union[pd.DataFrame, np.ndarray],
    names: Union[int, Iterable, None] = None,
    names_is: str = 'equal',
    ret_type: Union[str, None] = None,
    Xret_type: Union[pd.DataFrame, np.ndarray] = None,
    index=None,
    Xindex: Union[pd.DataFrame, np.ndarray] = None,
    Xnames: Union[pd.DataFrame, np.ndarray] = None,
    inplace=False,
):
    """
        Extract columns from DataFrame or ndarray, ordered by names (if it's a list)

        X :
            pd.DataFram or np.ndarray or list of tuples
        names : 
            list(str) -> list of names of the columns
            int -> number of columns
            None -> Don't care
        names_is :
            'subset' : check that names is a subset of the columns
            'superset' : check that names is a superset of the columns
            'equal' : check that names is exactly as the columns
        ret_type:
            None -> Same type as input
            'pd' -> returns a pandas.DataFrame
            'np' -> returns a numpy.ndarray    
            'dict'-> returns a list of dicts
            'tuple' -> return a list of tuples 
            'tf' -> return a tensor
        Xret_type :
            Object from which to extract the type (only if ret_type = None)
        index :
            Use this index if the return type is a pandas DataFrame
        Xindex : 
            Dataframe from which to extract the index (only if index = None)
        Xnames : 
            Dataframe from which to extract the names (only if names = None)
        
        TODO: names_is everywhere
    """

    # Handling of Series
    if isinstance(X, pd.Series):
        X = X.to_frame()

    input_type = type(X)

    if ret_type is None and Xret_type is not None:
        if isinstance(Xret_type, pd.DataFrame):
            ret_type = 'pd'
        elif isinstance(Xret_type, np.ndarray):
            ret_type = 'np'
        else:
            raise TypeError('Invalid type for Xret_type')
    if ret_type is None:
        ret_type = PROCESS_TYPE[input_type]

    if isinstance(names, Iterable):
        names = list(names)

    if names is None and isinstance(Xnames, pd.DataFrame):
        names = list(Xnames.columns.values)

    # DATAFRAME
    if input_type is pd.DataFrame:
        if isinstance(names, list):
            columns = frozenset(X.columns.values)
            if names_is == 'superset':
                # All the columns must be in names and are selected according to names order
                assert columns.issubset(frozenset(names)), f"names = {names} / columns = {X.columns.values}"

                # This is to order the returned columns by 'names'
                names = [name for name in names if name in columns]
            elif names_is == 'subset':
                # Selection of some columns based on names
                assert frozenset(names).issubset(columns), f"names = {names} / columns = {X.columns.values}"
            elif names_is == 'equal':
                assert frozenset(names) == columns, f"names = {names} / columns = {X.columns.values}"
            else:
                raise ValueError(f'Not supported names_is value ({names_is}) for this input type ({input_type})')

        elif isinstance(names, int):
            # Check the number of columns
            assert names == X.columns.values.shape[0]
            names = None
        if names is not None:
            X = X[names]

        if ret_type == 'pd':
            index = Xindex.index if index is None and isinstance(Xindex, pd.DataFrame) else index
            if index is not None:
                X = X.set_index(index)
            if inplace:
                return X
            else:
                return X.copy()
        elif ret_type == 'np':
            if inplace:
                return X.values
            else:
                return X.values.copy()
        elif ret_type == 'tf':
            return __to_tensor(X.values)
        elif ret_type == 'dict':
            return X.to_dict('records')
        elif ret_type == 'tuple':
            return list(X.itertuples(index=False, name=None))
        else:
            raise NotImplementedError(f'Not supported return type for input type: {input_type.__name__}')

    # NUMPY
    elif input_type is np.ndarray:
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        if isinstance(names, list):
            assert len(names) == X.shape[1], f"names = {names} and X.shape = {X.shape}"
        elif isinstance(names, int):
            assert names == X.shape[1]

        if ret_type == 'np':
            if inplace:
                return X
            else:
                return X.copy()
        elif ret_type == 'tf':
            return __to_tensor(X)
        elif ret_type == 'pd':
            index = Xindex.index if index is None and isinstance(Xindex, pd.DataFrame) else index
            return pd.DataFrame(X, columns=names, index=index)
        elif ret_type == 'dict':
            # names = names or [str(i) for i in range(names)]
            return [dict(zip(names, x)) for x in X]
        elif ret_type == 'tuple':
            return [tuple(x) for x in X]
        else:
            raise NotImplementedError(f'Not supported return type for input type: {input_type.__name__}')

    # LIST
    elif input_type is list:  # list of tuples
        element_type = type(X[0])

        # OF TUPLES, LISTS, or Numpy
        if element_type in [tuple, list, np.ndarray]:
            if isinstance(names, list):
                assert len(names) == len(X[0])
            elif isinstance(names, int):
                assert names == len(X[0])

            if ret_type == 'np':
                return np.array(X)
            elif ret_type == 'tf':
                return __to_tensor(np.array(X))
            elif ret_type == 'pd':
                index = Xindex.index if index is None and isinstance(Xindex, pd.DataFrame) else index
                return pd.DataFrame(X, columns=names, index=index)
            elif ret_type == 'dict':
                # names = names or [str(i) for i in range(names)]
                return [dict(zip(names, x)) for x in X]
            elif ret_type == 'tuple':
                if inplace and element_type is tuple:
                    return X
                else:
                    return [tuple(x) for x in X]
            else:
                raise NotImplementedError(f'Not supported return type for input type: {input_type.__name__}')
        else:
            raise NotImplementedError(f'Not supported input type: {input_type.__name__}[{element_type.__name__}]')

    else:
        raise TypeError(f'Not supported input type: {input_type.__name__}')


# Alias
process_data = data_astype

# def concatenate_data(A: Union[pd.DataFrame, np.ndarray, list], B: Union[pd.DataFrame, np.ndarray, list]):
#     if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
#         return pd.concat([A, B], axis=1)
#     elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
#         A = A.reshape(-1, 1) if len(A.shape) == 1 else A
#         B = B.reshape(-1, 1) if len(B.shape) == 1 else B
#         return np.concatenate([A, B], axis=1)
#     else:
#         raise TypeError('Can handle only pandas and numpy.')
