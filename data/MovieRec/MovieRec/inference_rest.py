import torch
import pickle
from pathlib import Path
from models import BERTModel
from model_args import args
import pandas as pd

with open('id_2_real_id.pickle', 'rb') as handle:
    mapping = pickle.load(handle)
u_2_real_u, movie_2_real_movie = mapping['u_2_real_u'], mapping['movie_2_real_movie']


movie_info = pd.read_csv('/workspaces/MovieRec/Data/ml-20m/movies.csv')  
movie_info.columns = ['movieId', 'title', 'genres']

item_size = len(movie_2_real_movie)
args.num_items = item_size
masked_token = item_size + 1

best_model_path = '/workspaces/MovieRec/experiments/test_2022-03-14_0/models/best_acc_model.pth'


history = torch.randint(0, item_size, (1, args.bert_max_len-1), dtype=torch.long)
seq = torch.cat((history, torch.tensor(masked_token).reshape(1, 1)), dim=1).to(device=args.device)

print(seq)

model = BERTModel(args).to(device=args.device)
best_model = torch.load(best_model_path).get('model_state_dict')
model.load_state_dict(best_model)
model.eval()

scores = model(seq)
movie_id = scores[:, -1, :].argmax(dim=-1).to('cpu').item()
print(movie_id)
real_movie_id = movie_2_real_movie[movie_id]
movie = movie_info[movie_info['movieId'] == real_movie_id].iloc[0]
title, genres = movie['title'], movie['genres']
print(f'{real_movie_id}, {title}, {genres}')