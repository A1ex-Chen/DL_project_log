def main():
    args = parseArgs()
    pyprof.nvtx.init()
    pyprof.nvtx.wrap(fused_adam_cuda, 'adam')
    N = args.b
    C = 3
    H = d[args.m]['H']
    W = d[args.m]['W']
    opts = d[args.m]['opts']
    classes = 1000
    net = getattr(models, args.m)
    net = net(**opts).cuda().half()
    net.train()
    x = torch.rand(N, C, H, W).cuda().half()
    target = torch.empty(N, dtype=torch.long).random_(classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    if args.o == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.o == 'adam':
        optimizer = FusedAdam(net.parameters())
        optimizer = FP16_Optimizer(optimizer)
    else:
        assert False
    for i in range(2):
        output = net(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.autograd.profiler.emit_nvtx():
        profiler.start()
        output = net(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        profiler.stop()
