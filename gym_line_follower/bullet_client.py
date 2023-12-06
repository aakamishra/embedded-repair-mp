import functools
import inspect
import pybullet
from functools import wraps

def add_optional_argument(arg_name, default_value):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            kwargs[arg_name] = default_value
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        if self._client < 0:
            print("options=", options)
            self._client = pybullet.connect(connection_mode, options=options)
        self._shapes = {}
        super().__init__()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and not attr_name.startswith("__"):  # Exclude special methods
                setattr(self, attr_name, add_optional_argument('optional_arg', None)(attr))


    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute
