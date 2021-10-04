#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start UI.

Start Crcordile with an UI.
"""
import sys

from dialog import Dialog

import crocrodile
from crocrodile.client import main as client_main


class StartUI(object):
    """Crocrodile Start UI."""
    def __init__(self):
        self.dialog: Dialog = Dialog()
        self.dialog.set_background_title("Crocrodile Start Program")

    def main_menu(self) -> int:
        """
        Show main menu.

        :return: Choice (1 client, 2 basics train)
        :rtype: int
        """
        code, tag = self.dialog.menu("Wich element do you want to start ?",
                                     title="Main menu",
                                     choices=[("1", "Client"),
                                              ("2", "Basics Training")])
        if code == self.dialog.OK:
            return int(tag)

    def input_client_options(self) -> list:
        """
        Input client options.

        :return: List of selected arguments.
        :rtype: list
        """
        code, tags = self.dialog.checklist("Select options of client:",
                                           choices=[("-d", "Use developpement account", False),
                                                    ("-n", "Use neural network", False),
                                                    ("-u", "Upgrade account to BOT", False),
                                                    ("-v", "Show debug logs", False),
                                                    ("-q", "Don't show any logs", False),
                                                    ("-c", "Challenge user", False),
                                                    ("-a", "Auto re-challenging", False)],
                                           title="Configure client")
        if code == self.dialog.OK:
            return tags

    def ask_challenge(self) -> list:
        """
        Ask challenge configuration.

        :return: List of options.
        :rtype: list
        """
        code, return_list = self.dialog.form("Select challenge configuration:",
                                      [("User", 1, 1, "", 1, 1, 0, 0)])

    def ask_challenge_user(self) -> str:
        """
        Ask user to challenge.

        :return: Username of user to challenge.
        :rtype: str
        """
        code, string = self.dialog.inputbox("User to challenge:", title="Configure client")
        if code == self.dialog.OK:
            return string

    def ask_time_control(self) -> str:
        """
        Ask time control for challenge.

        :return: Time control (like 5+3).
        :rtype: str
        """
        code, string = self.dialog.inputbox("Time control for challenge:", title="Configure client")
        if code == self.dialog.OK:
            return string

    def start(self) -> None:
        """
        Start UI.

        :return: Nothing.
        :rtype: None
        """
        choice: int = self.main_menu()
        if choice == 1:  # Start client
            self.start_client()
        elif choice == 2:  # Start basics training
            self.start_basics()

    def start_client(self) -> None:
        """
        Start client.

        :return: Nothing.
        :rtype: None
        """
        options: list = self.input_client_options()
        if "-c" in options:
            form_result: list = self.ask_challenge()
            user: str = self.ask_challenge_user()
            time_control: list = self.ask_time_control().split("+")
            if len(time_control) != 2:
                self.challenge_invalid()
            try:
                time: int = int(time_control[0])
                increment: int = time_control[1]
            except ValueError:
                self.challenge_invalid()
                return
            args = f"{user} {time} {increment}"
            options.append(args)
        return client_main(options)

    def challenge_invalid(self) -> str:
        """
        Show invalid challenge configuration.
        
        :return: Dialog exit code.
        :rtype: str
        """
        return self.dialog.msgbox("ERROR: Invalid challenge", title="Configure client")


if __name__ == "__main__":
    ui = StartUI()
    sys.exit(ui.start())
