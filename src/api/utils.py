import time
import random
import logging


def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 60,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = tuple(),
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay: float = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                    delay *= min(exponential_base * (1 + jitter * random.random()), max_delay)
                    time.sleep(delay)
                    logging.error(msg=f"Retrying #{num_retries}...")
                except Exception as e:
                    raise e

        return wrapper

    return decorator
