from functools import wraps


def catch_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            return e

    return wrapper
