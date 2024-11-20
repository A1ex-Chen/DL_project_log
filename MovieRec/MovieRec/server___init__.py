def __init__(self):
    dataset = pickle.load(Path(
        '/workspaces/MovieRec/Data/preprocessed/ml-20m_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl'
        ).open('rb'))
    self.real_u_2_u = dataset['umap']
    self.real_movie_2_movie = dataset['smap']
    with open('id_2_real_id.pickle', 'rb') as handle:
        mapping = pickle.load(handle)
        self.u_2_real_u, self.movie_2_real_movie = mapping['u_2_real_u'
            ], mapping['movie_2_real_movie']
    self.movie_info = pd.read_csv('/workspaces/MovieRec/Data/ml-20m/movies.csv'
        )
    self.movie_info.columns = ['movieId', 'title', 'genres']
    item_size = len(self.movie_2_real_movie)
    print(f'item_size: {item_size}')
    args.num_items = item_size
    self.seq_len = args.bert_max_len
    self.masked_token = item_size + 1
    best_model_path = (
        '/workspaces/MovieRec/experiments/test_2022-03-14_0/models/best_acc_model.pth'
        )
    model = BERTModel(args).to(device=args.device)
    best_model = torch.load(best_model_path).get('model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    self.model = model
    self.device = args.device
    posters_file = []
    for im in os.listdir('/workspaces/MovieRec/MLP-20M'):
        real_id = int(im.split('.')[0])
        if real_id in self.real_movie_2_movie:
            posters_file.append(real_id)
    self.movie_with_posters = posters_file
