def initialize_factors(self, train_data, k=25):
    self.ratings = train_data[['user_id', 'item_id', 'rating']].as_matrix()
    self.k = k
    self.user_ids = pd.unique(train_data['user_id'])
    self.item_ids = pd.unique(train_data['item_id'])
    self.u_inx = {r: i for i, r in enumerate(self.user_ids)}
    self.i_inx = {r: i for i, r in enumerate(self.item_ids)}
    self.user_factors = np.random.random_sample((len(self.user_ids), k))
    self.item_factors = np.random.random_sample((len(self.item_ids), k))
    self.user_item = train_data.groupby('user_id')['item_id'].apply(lambda
        x: x.tolist()).to_dict()
    self.item_bias = defaultdict(lambda : 0)
    self.create_loss_samples()
