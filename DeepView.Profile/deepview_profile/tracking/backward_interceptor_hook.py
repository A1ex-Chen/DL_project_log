def hook(*args):
    self.backward_root = args[0]
    raise _SuspendExecution
