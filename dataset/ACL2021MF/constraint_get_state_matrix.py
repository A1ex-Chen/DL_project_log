def get_state_matrix(self, output_size, constraints, image_id):
    assert len(constraints) <= self.max_constrain_num
    M = cbs_matrix(output_size)
    M.init_matrix(self.state_size)
    self._num_cls[image_id] = len(constraints)
    con_str = []
    for c in constraints:
        c_list = ['#'.join([str(i) for i in x]) for x in c]
        con_str.append('^'.join(c_list))
    marker = '*'.join(con_str) if len(con_str) > 0 else '***'
    if marker not in self._cache:
        comb_constrains = []
        for c in constraints:
            comb_constrains += c
        additional_state = len(constraints) + 1
        for i in range(additional_state):
            M.init_row(i)
        for i in range(len(constraints)):
            M, additional_state = self.connect_edge(M, additional_state, i,
                i + 1, comb_constrains)
        self._cache[marker] = M.get_matrix(), additional_state
    return self._cache[marker]
