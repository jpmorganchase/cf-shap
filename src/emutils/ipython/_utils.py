import warnings

__all__ = [
    'in_ipynb',
    'display',
    'display_title',
    'end',
    'import_tqdm',
    'notebook_fullwidth',
]


def in_ipynb():
    '''
        Check if we are in a IPython Environment
        Returns True if in IPython, or False otherwise
    '''
    # pylint: disable=import-outside-toplevel
    # pylint: disable=expression-not-assigned
    # pylint: disable=bare-except

    try:
        from IPython.core.getipython import get_ipython
        get_ipython().config
        return True
    except:  # noqa: E722
        return False
    # pylint: enable=import-outside-toplevel
    # pylint: enable=expression-not-assigned
    # pylint: enable=bare-except


def display(*args, **kwargs):
    if in_ipynb():
        try:
            from IPython.display import display as ipython_display
            return ipython_display(*args, **kwargs)
        except:  # noqa: E722

            warnings.warn("ERROR loading display from IPython")
            pass


def display_title(title, style='h2'):
    if in_ipynb():
        from IPython.core.display import display, HTML
        display(HTML(f'<{style}>{title}</{style}>'))
    else:
        print(title.upper())


def end(
    message: str = 'This is the END.',
    only_others: bool = False,
):
    """
    Stop the execution with a message:
        - Raising an exception in Jupyter/IPython environment
        - Quitting in other environments

    Args:
        message (str, optional): Message that is displayed. Defaults to 'This is the END.'.
        only_others (bool, optional): Ends exeution only in other environments. Defaults to False.
    """
    if in_ipynb():
        if not only_others:
            raise Exception(message)
    else:
        print(message)
        quit()


def import_tqdm():
    # pylint: disable=import-outside-toplevel
    if in_ipynb():
        try:
            from tqdm.notebook import tqdm
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Old version of TQDM
            from tqdm import tqdm
    else:
        from tqdm import tqdm
    # pylint: enable=import-outside-toplevel
    return tqdm


def notebook_fullwidth():
    """Set the Jupyter Notebook width to the maximum page width
    """
    if in_ipynb():
        # pylint: disable=import-outside-toplevel
        from IPython.core.display import display, HTML
        # pylint: enable=import-outside-toplevel

        display(HTML("<style>.container { width:100% !important; }</style>"))
