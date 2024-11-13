import sys

from foc import *
from ouch import *


def executable(cmd, script):
    if exists(cmd):
        prompt(
            f"'{cmd}' already exists. Are you sure to proceed?",
            fail=lazy(error, f"\n.Operation aborted, '{cmd}'"),
        )
    f = writer(cmd)
    f.write("#!/bin/bash\n")
    f.write(f'python {script} "$@"\n')
    o = shell(f"chmod 755 {cmd}")
    if not o:
        print(f"generated {cmd}")


if __name__ == "__main__":
    guard(
        len(sys.argv) == 2,
        "Invalid number of arguments. Provide a command name.",
    )
    cmd = f"{HOME()}/.local/bin/{sys.argv[-1]}"
    script = f" {dirname(__file__)}/cli.py"
    executable(cmd, script)
