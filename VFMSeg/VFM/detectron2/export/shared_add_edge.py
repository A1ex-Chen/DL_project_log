def add_edge(self, u, v):
    self.graph[u].append(v)
    self.vertices.add(u)
    self.vertices.add(v)
