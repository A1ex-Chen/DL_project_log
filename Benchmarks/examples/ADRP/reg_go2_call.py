def call(self, V):
    Q = ke.backend.dot(V, self.kernel)
    Q = Q * V
    Q = Q / math.sqrt(self.output_dim)
    Q = ke.activations.softmax(Q)
    return Q
