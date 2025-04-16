def __exit__(self, exc_type, value, traceback):
    if value:
        print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
    return True
