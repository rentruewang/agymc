import subprocess as sp
import sys

if __name__ == "__main__":
    versions = sys.argv[1:]
    versions = dict(entry.split("=") for entry in packages)

    pip = "env python3 -m pip"
    with sp.Popen(
        f"{pip} freeze", shell=True, stdout=sp.PIPE, encoding="utf-8"
    ) as pipe:
        packages = str(pipe.stdout.read()).strip("\n").split("\n")
        packages = dict(entry.split("==") for entry in packages)
