#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show a graph of NNs logs."""
import os
import sys

import matplotlib.pyplot as plt


def show_graph(path_to_log_file: str) -> None:
    """
    Show graph from NNs log file.

    :param str path_to_log_file: Path to the NN log file.
    :return: None
    :rtype: None
    """
    print("Showing", path_to_log_file)
    with open(path_to_log_file) as file:
        content: list[str] = file.read().split("\n")
        content.pop()

    values: list = list()
    for string in content:
        values.append(float(string))

    plt.plot(values)
    plt.axis(ymin=20, ymax=40)
    return plt.show()


def main() -> int:
    """
    Main function.

    :return: Exit code.
    :rtype: int
    """
    directory_content: list = os.listdir("nns/log")
    elements: dict[int] = {}
    for index, element in enumerate(directory_content):
        elements[index] = element
        print(f"{index + 1}) {element}")

    while True:
        try:
            choice: int = int(input("Log to show: "))
            if 0 < choice <= len(directory_content):
                break
            else:
                print("Please enter a number between 0 and", len(directory_content))
        except ValueError:
            print("Please enter a valid number")
    
    show_graph("nns/log/" + elements[choice - 1])


if __name__ == "__main__":
    sys.exit(main())
