def test_gpu(gpu_id=[0]):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        device = torch.device('cuda:%s' % gpu_id[0])
    else:
        device = torch.device('cpu')
    pose_size = 72
    beta_size = 300
    np.random.seed(9608)
    model = SMPLModel(device=device, model_path=
        '/remote-home/my/model_300_m.pkl')
    for i in range(10):
        pose = torch.from_numpy((np.random.rand(8, pose_size) - 0.5) * 0.4
            ).type(torch.float64).to(device)
        betas = torch.from_numpy((np.random.rand(8, beta_size) - 0.5) * 0.06
            ).type(torch.float64).to(device)
        s = time()
        trans = torch.from_numpy(np.zeros((8, 3))).type(torch.float64).to(
            device)
        result, joints = model(betas, pose, trans)
        print(time() - s)
