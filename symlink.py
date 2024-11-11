from foc import *
from ouch import *


def symlink(path):
    if exists(path):
        prompt(
            f"'{path}' already exists. Are you sure to proceed?",
            fail=lazy(error, f"\nAborted. Won't overwrite '{path}'."),
        )
    f = writer(path)
    f.write("#!/bin/bash\n")
    f.write(f'python {pwd()}/cli.py "$@"\n')
    shell(f"chmod 755 {path}")


if __name__ == "__main__":
    path = f"{HOME()}/.local/bin/voh"
    symlink(path)
