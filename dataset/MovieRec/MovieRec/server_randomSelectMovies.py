def randomSelectMovies(self, count=2000):
    random_idx = np.random.choice(len(self.movie_with_posters), count,
        replace=False).tolist()
    movies = []
    ids = []
    for i in random_idx:
        id = self.movie_with_posters[i]
        movie = self.movie_info[self.movie_info['movieId'] == id]
        if movie.shape[0] > 0:
            title, genres = movie.iloc[0]['title'], movie.iloc[0]['genres']
            movies.append({'id': id, 'title': title, 'genres': genres})
            ids.append(id)
    return {'movies': movies}
