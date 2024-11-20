def get_k_neighbor(self, username):
    distances = []
    for instance in self.data:
        if instance != username:
            distance = self.fn(self.data[username], self.data[instance])
            distances.append((instance, distance))
    distances.sort(key=lambda cur_bookTuple: cur_bookTuple[1], reverse=True)
    return distances
