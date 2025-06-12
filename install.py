import contextlib
import os

path = os.path.dirname(os.path.realpath(__file__))

# we remove __init__ because it breaks BLIP - takes over the directory named models which BLIP also uses.
with contextlib.suppress(OSError):
    os.remove(os.path.join(path, "Pixelization", "models", "__init__.py"))
