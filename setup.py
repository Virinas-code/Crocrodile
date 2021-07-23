#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Setup.

Setup Crocrodile chess engine.
"""
import sys
import os
import pkgutil
import requests


def load_requirements(requirements):
    """Load requirements."""
    to_install = requirements
    installed = list()
    for module in pkgutil.iter_modules():
        if module.name in to_install:
            to_install.remove(module.name)
            installed.append(module.name)
    return to_install, installed


def download(url, destination):
    request = requests.get(url, stream=True)
    request.raw.decode_content = True
    with open(destination, 'wb') as destination:
        destination.write(request.raw.read())


def detect_python():
    """Detect Python installation."""
    print("Detecting Python installation...", end=" ", flush=True)
    test = os.system("python3 --version > " + os.devnull)
    if test == 0:
        print("Done.")
        return "python3"
    test = os.system("py -3 --version > " + os.devnull)
    if test == 0:
        print("Done.")
        return "py -3"
    print("Done.")
    print("/!\\ Python installation not found !")
    print("    Please verify that Python is installed and launchable with 'python3' command")
    stop()


def detect_pip(python):
    """Detect pip installation."""
    print("Detecting pip installation...", end=" ", flush=True)
    test = os.system(python + " -m pip --version > " + os.devnull)
    print("Done.")
    if test == 0:
        return
    print("Downloading: https://bootstrap.pypa.io/get-pip.py...", end=" ", flush=True)
    download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
    print("Done.")
    print("Installing: pip...", end=" ", flush=True)
    test = os.system(python + " get-pip.py > " + os.devnull)
    print("Done.")
    if test != 0:
        print("/!\\ Failed to install pip.")
        stop()


def install_requirements(requirements, python):
    """Install missing requirements."""
    for requirement in requirements:
        print("Downloading: https://pypi.org/simple/{0}/...".format(requirement), end=" ", flush=True)
        for retry in range(3):
            test = os.system(python + " -m pip download --only-binary=:all: " + requirement + " > " + os.devnull)
            if test == 0:
                break
            if retry == 0:
                print()
            print("Failed to download, retrying (" + str(retry + 1) + "/3)")
        if retry == 2:
            print("\n/!\\ Failed to download", requirement)
            stop()
        print("Done.")
    --no-dependencies
    for wheel in glob.glob("*.whl"):
        print("Installing:", wheel + "...", end=" ", flush=True)
        test = os.system(python + " -m pip install --no-dependencies " + wheel + " > " + os.devnull)
        if test != 0:
            print("\n/!\\ Failed to install", wheel)
        print("Done.")


def install(requirements):
    """Install Crocrodile."""
    python = detect_python()
    detect_pip(python)
    install_requirements(requirements, python)


def stop():
    """Stop installation."""
    print("Installation canceled.")
    sys.exit(1)


requirements = ["berserk", "python-chess", "colorama", "pip"]


print("Welcome in Crocrodile setup program !")

print("Loading requirements...", end=" ", flush=True)
to_install, installed = load_requirements(requirements)
print("Done.")

print()
print("Installed packages:")
print("   ", "\t".join(installed))
print("Packages to install:")
print("   ", "\t".join(to_install))
continuation = input("Continue ? [Y/n] ").lower()
if continuation in ("y", "yes", ""):
    install(to_install)
elif continuation in ("n", "no"):
    print("Canceling...")
    stop()
else:
    print("Bad answer. Canceling...")
    stop()
