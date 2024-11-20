def print_tensor_test(tensor, limit_to_slices=None, max_torch_print=None,
    filename='test_corrections.txt', expected_tensor_name='expected_slice'):
    if max_torch_print:
        torch.set_printoptions(threshold=10000)
    test_name = os.environ.get('PYTEST_CURRENT_TEST')
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)
    if limit_to_slices:
        tensor = tensor[0, -3:, -3:, -1]
    tensor_str = str(tensor.detach().cpu().flatten().to(torch.float32)
        ).replace('\n', '')
    output_str = tensor_str.replace('tensor',
        f'{expected_tensor_name} = np.array')
    test_file, test_class, test_fn = test_name.split('::')
    test_fn = test_fn.split()[0]
    with open(filename, 'a') as f:
        print('::'.join([test_file, test_class, test_fn, output_str]), file=f)
