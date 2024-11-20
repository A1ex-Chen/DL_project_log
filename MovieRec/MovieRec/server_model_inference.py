def model_inference(self, real_movie_seq):
    assert len(real_movie_seq) == self.seq_len - 1
    history = torch.tensor([self.real_movie_2_movie[s] for s in
        real_movie_seq], dtype=torch.long).reshape(1, self.seq_len - 1)
    seq = torch.cat((history, torch.tensor(self.masked_token).reshape(1, 1)
        ), dim=1).to(device=self.device)
    scores = self.model(seq)
    movie_id = scores[:, -1, :].argmax(dim=-1).to('cpu').item()
    real_movie_id = self.movie_2_real_movie[movie_id]
    movie = self.movie_info[self.movie_info['movieId'] == real_movie_id].iloc[0
        ]
    title, genres = movie['title'], movie['genres']
    print(f'Next recommended movie: {real_movie_id}, {title}, {genres}')
    return {'title': title, 'id': real_movie_id, 'genres': genres}
