"""Boostrap"""

import os
import sys
import shutil
import platform


def main():
    """Console bootstrap to handle different OS

    - MacOS: if available, re-exec under `mjpython` + run `farms_sim.farmsim`.
    - Else: import and run the real CLI directly.

    """

    # Prevent loops if already under mjpython
    if os.environ.get("FARMS_UNDER_MJPYTHON") != "1" and platform.system() == "Darwin":
        mj = shutil.which("mjpython")
        if mj:
            env = os.environ.copy()
            env["FARMS_UNDER_MJPYTHON"] = "1"
            os.execvpe(mj, [mj, "-m", "farms_sim.farmsim", *sys.argv[1:]], env)
        else:
            print(
                "[FARMS] macOS detected but `mjpython` not found in PATH; "
                "continuing with current Python (MuJoCo may fail to load).",
                file=sys.stderr,
            )

    # Only import your CLI after deciding about re-exec
    from .farmsim import profile_simulation
    profile_simulation()
