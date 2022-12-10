def get_function_from_signature(signature):
    """Parases a function of the form module.submodule:function to import
    and get the actual function as defined.
    
    Parameters
    ----------
    signature : str
    
    Returns
    -------
    callable
    """

    module, function = signature.split(":")
    eval(f"from {module} import {function}")
    return eval(function)
