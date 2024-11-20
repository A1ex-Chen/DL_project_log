@property
def callback_list(self):
    return '\n'.join(cb.__class__.__name__ for cb in self.callbacks)
