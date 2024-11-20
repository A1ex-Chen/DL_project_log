def filterTrace(mlist):
    """
			Filter trace markers to remove certain file names.
			"""
    assert type(mlist) == list
    if len(mlist) == 0:
        return mlist
    mlist = mlist[-1]
    mlist = eval(mlist)
    mlist = mlist['traceMarker']
    assert type(mlist) == list
    mlist = list(filter(lambda x: '/torch/nn/modules/' not in x, mlist))
    mlist = list(filter(lambda x: '/torch/nn/functional.py' not in x, mlist))
    mlist = list(filter(lambda x: '/torch/tensor.py' not in x, mlist))
    mlist = list(filter(lambda x: '/torch/autograd/__init__.py' not in x,
        mlist))
    mlist = list(filter(lambda x: '/torch/_jit_internal.py' not in x, mlist))
    mlist = list(filter(lambda x: '/pyprof/nvtx/nvmarker.py' not in x, mlist))
    mlist = list(filter(lambda x: '/apex/optimizers/' not in x, mlist))
    mlist = list(filter(lambda x: '/torch/_utils.py' not in x, mlist))
    mlist = list(filter(lambda x: '/torch/optim/' not in x, mlist))
    return mlist
