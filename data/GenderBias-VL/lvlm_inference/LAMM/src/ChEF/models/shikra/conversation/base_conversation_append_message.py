def append_message(self, role: str, message: str):
    """Append a new message."""
    self.messages.append([role, message])
