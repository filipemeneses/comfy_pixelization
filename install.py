import contextlib
import os
import subprocess

path = os.path.dirname(os.path.realpath(__file__))

def git(*args):
    return subprocess.check_call(["git", *list(args)])

# Install Pixelization submodule
git("submodule", "update", "--init")

# we remove __init__ because it breaks BLIP - takes over the directory named models which BLIP also uses.
with contextlib.suppress(OSError):
    os.remove(os.path.join(path, "Pixelization", "models", "__init__.py"))
