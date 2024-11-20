def __init__(self, molecular_model, molecular_encoder, files, mb_epochs,
    callbacks, save_path='.', batch_size=32, nbr_type='relative',
    len_molecular_hidden_layers=1, molecular_nbrs=0, conv_bool=False,
    full_conv_bool=False, type_bool=False, sampling_density=1.0):
    self.files = files
    self.molecular_model = molecular_model
    self.molecular_encoder = molecular_encoder
    self.mb_epochs = mb_epochs
    self.callbacks = callbacks
    self.nbr_type = nbr_type
    self.batch_size = batch_size
    self.len_molecular_hidden_layers = len_molecular_hidden_layers
    self.molecular_nbrs = molecular_nbrs
    self.conv_net = conv_bool or full_conv_bool
    self.full_conv_net = full_conv_bool
    self.type_feature = type_bool
    self.save_path = save_path + '/'
    self.sampling_density = sampling_density
    self.test_ind = random.sample(range(len(self.files)), 1)
    self.train_ind = np.setdiff1d(range(len(self.files)), self.test_ind)
