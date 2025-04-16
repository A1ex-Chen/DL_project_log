def __init__(self, device=None, model_path='./model.pkl'):
    super(SMPLModel, self).__init__()
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].
        todense())).type(torch.float64)
    if 'joint_regressor' in params.keys():
        self.joint_regressor = torch.from_numpy(np.array(params[
            'joint_regressor'].T.todense())).type(torch.float64)
    else:
        self.joint_regressor = torch.from_numpy(np.array(params[
            'J_regressor'].todense())).type(torch.float64)
    self.weights = torch.from_numpy(params['weights']).type(torch.float64)
    self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64)
    self.v_template = torch.from_numpy(params['v_template']).type(torch.float64
        )
    self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float64)
    self.kintree_table = params['kintree_table']
    self.faces = params['f']
    self.device = device if device is not None else torch.device('cpu')
    for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs',
        'v_template', 'shapedirs']:
        _tensor = getattr(self, name)
        setattr(self, name, _tensor.to(device))
