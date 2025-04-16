def common_init(test_case):
    test_case.h = 64
    test_case.b = 16
    test_case.c = 16
    test_case.k = 3
    test_case.t = 10
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
