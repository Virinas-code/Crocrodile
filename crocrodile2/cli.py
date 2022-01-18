def start(*args, **kwargs):
    print(*args, end="... ", **kwargs)


def done():
    print("Done.")


class Progress:
    def __init__(self):
        self.total = None
        self.text = "Loading"
        self.update()

    def update(self, value=None):
        if value is None:
            if self.total is None:
                print(self.text + "... (initializing)", end="\r")
            else:
                print(self.text + "...", end="\r")
        else:
            if self.total is None:
                print(self.text + f"... ({value})", end="\r")
            else:
                print(self.text + f"... ({value}/{self.total})", end="\r")

    def done(self):
        print(f"{self.text}... Done.           ")
