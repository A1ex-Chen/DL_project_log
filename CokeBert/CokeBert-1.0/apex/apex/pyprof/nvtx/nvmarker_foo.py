def foo(args, name):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.dim() == 0:
                scalar(arg.item(), name)
            else:
                tensor(arg, name)
        elif isinstance(arg, numpy.ndarray):
            ndarray(arg, name)
        elif isscalar(arg):
            scalar(arg, name)
        elif issequence(arg):
            if len(arg) == 0 or isscalar(arg[0]):
                seq(arg, name)
            else:
                foo(arg, name)
        """
			else:
				print("The following arg is none of Tensor, numpy array, scalar but a %s" % (str(type(arg))))
				print("Mod: %s" % str(mod.__name__))
				print("Op: %s" % str(op))
				print(dir(arg))
			"""
