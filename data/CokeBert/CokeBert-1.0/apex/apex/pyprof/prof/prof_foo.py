def foo(mod, op, d):
    if op[0] == 'linear':
        xx = Linear(d)
    elif mod[0] in ['LSTMCell', 'GRUCell'] and op[0] == 'forward':
        xx = RNNCell(d)
    elif op[0] in ['conv1d', 'conv2d']:
        xx = Conv(d)
    elif op[0] in Pointwise.ops:
        xx = Pointwise(d)
    elif op[0] in Convert.ops:
        xx = Convert(d)
    elif op[0] in ['__matmul__', 'matmul']:
        xx = Matmul(d)
    elif op[0] == 'embedding':
        xx = Embedding(d)
    elif op[0] == 'sum':
        xx = Sum(d)
    elif op[0] == 'mean':
        xx = Mean(d)
    elif op[0] == 'norm':
        xx = Norm(d)
    elif op[0] == 'dropout':
        xx = Dropout(d)
    elif op[0] == 'cat':
        xx = Cat(d)
    elif op[0] == 'reshape':
        xx = Reshape(d)
    elif op[0] == 'masked_scatter_':
        xx = MaskedScatter(d)
    elif op[0] == 'gather':
        xx = Gather(d)
    elif op[0] == 'nonzero':
        xx = Nonzero(d)
    elif op[0] == 'index_select':
        xx = IndexSelect(d)
    elif op[0] == 'masked_select':
        xx = MaskedSelect(d)
    elif op[0] in ['addmm', 'addmm_']:
        xx = Addmm(d)
    elif op[0] == 'mm':
        xx = Mm(d)
    elif op[0] == 'bmm':
        xx = Bmm(d)
    elif op[0] == 'softmax':
        xx = Softmax(d)
    elif op[0] == 'log_softmax':
        xx = LogSoftmax(d)
    elif op[0] == 'mse_loss':
        xx = MSELoss(d)
    elif op[0] == 'adam':
        xx = Adam(d)
    elif op[0] == 'batch_norm':
        xx = BatchNorm(d)
    elif op[0] == 'randperm':
        xx = RandPerm(d)
    elif op[0] == 'copy_':
        xx = Copy(d)
    elif op[0] == 'clone':
        xx = Clone(d)
    elif op[0] == 'contiguous':
        xx = Contiguous(d)
    elif op[0] == 'any':
        xx = Any(d)
    elif op[0] in Activation.ops:
        xx = Activation(d)
    elif op[0] == 'to':
        xx = Convert(d)
    else:
        xx = Foo(d)
    return xx
