"""
    Author: Emanuele Albini

    This module contains general utilities for Python programming language.
    - Extension and improvement of Python data structures (e.g., attrdict)
    - Extension of Python language (e.g., static variables for functions, function caching)
    - Implementations of design patterns in Python (e.g., Singleton)
"""

from abc import ABCMeta
from collections import defaultdict
from collections.abc import Mapping
from pprint import pformat
import json

__all__ = [
    'static_vars',
    'keydefaultdict',
    'ignorenonedict',
    'attrdict',
    'function_cache',
    'Singleton',
]


def static_vars(**kwargs):
    """Decorator to create static variables for a function

    Usage:
    ```python
    @static_vars(i = 0)
    def function(x):
        i += 1
        return x + i

    function(0) # 1
    function(0) # 2
    function(10) # 13
    ```
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


class keydefaultdict(defaultdict):
    """
        Extension of defaultdict that support
        passing the key to the default_factory
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key, "Must pass a default factory with a single argument.")
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class ignorenonedict(dict):
    def __init__(self, other=None, **kwargs):
        super().__init__()
        self.update(other, **kwargs)

    def __setitem__(self, key, value):
        if value is not None:
            super().__setitem__(key, value)

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self.__setitem__(k, v)
        for k, v in kwargs.items():
            self.__setitem__(k, v)


class attrdict(dict):
    """
        Attributes-dict bounded structure for paramenters
        -> When a dictionary key is set the corresponding attribute is set
        -> When an attribute is set the corresponding dictionary key is set

        Usage:

            # Create the object
            args = AttrDict()
            
            args.a = 1
            print(args.a) # 1
            print(args['a']) # 1

            args['b'] = 2
            print(args.b) # 2
            print(args['b']) # 2

    """
    def __init__(self, *args, **kwargs):
        super(attrdict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def repr(self):
        return dict(self)

    def __repr__(self):
        return pformat(self.repr())

    def __str__(self):
        return self.__repr__()

    def update_defaults(self, d: dict):
        for k, v in d.items():
            self.setdefault(k, v)

    def save_json(self, file_name):
        with open(file_name, 'w') as fp:
            json.dump(self.repr(), fp)

    def copy(self):
        return type(self)(self)


class FunctionCache(dict):
    def __init__(self, f):
        self.f = f

    def get(self, *args, **kwargs):
        key = hash(str(args)) + hash(str(kwargs))
        if key in self:
            return self[key]
        else:
            ret = self[key] = self.f(*args, **kwargs)
            return ret


# Decorator
def function_cache(f, name='cache'):
    """"
    
    Usage Example:

    @function_cache(lambda X: expensive_function(X))
    @function_cache(lambda X: expensive_function2(X), name = 'second_cache')
    def f(X, y):
        return expensive_function(Y) - f.cache.get(X) + f.second_cache.get(X)


    X, Y = ..., ...
    for y in Y:
        f(X, y) # The function is called multiple times with X



    """
    def decorate(func):
        setattr(func, name, FunctionCache(f))
        return func

    return decorate


class cached_property(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")

        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)

        return value


class Singleton(ABCMeta):
    """
    Singleton META-CLASS from https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    Usage Example:
    class Logger(metaclass=Singleton):
        pass

    What it does:
    When you call logger with Logger(), Python first asks the metaclass of Logger, Singleton, what to do, 
    allowing instance creation to be pre-empted. This process is the same as Python asking a class what to do 
    by calling __getattr__ when you reference one of it's attributes by doing myclass.attribute.

    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]