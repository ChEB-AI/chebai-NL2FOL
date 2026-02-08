from functools import wraps


def tptp_parse_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            return e

    return wrapper


def mol_to_fol_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            return e

    return wrapper


def model_check_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            return e

    return wrapper
