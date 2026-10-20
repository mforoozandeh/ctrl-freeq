"""Make one nutate call in a child process (see ``plotter._run_nutate``).

Usage: ``python -P _nutate_child.py CALL`` where *CALL* is a pickled
``(name, args, kwargs)`` for ``nutate.<name>(*args, **kwargs)``.
"""

import pickle
import sys

import nutate as nt

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        name, args, kwargs = pickle.load(f)
    getattr(nt, name)(*args, **kwargs)
