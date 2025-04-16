def forward(self, betas, pose, trans, simplify=False):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.
          
          20190128: Add batch support.

          Parameters:
          ---------
          pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [N, 3].

          Return:
          ------
          A 3-D tensor of [N * 6890 * 3] for vertices,
          and the corresponding [N * 19 * 3] joint positions.

    """
    batch_num = betas.shape[0]
    id_to_col = {self.kintree_table[1, i]: i for i in range(self.
        kintree_table.shape[1])}
    parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1, self
        .kintree_table.shape[1])}
    v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])
        ) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(
        batch_num, -1, 3, 3)
    if simplify:
        v_posed = v_shaped
    else:
        R_cube = R_cube_big[:, 1:, :, :]
        I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) +
            torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.
            float64)).to(self.device)
        lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=(
            [1], [2]))
    results = []
    results.append(self.with_zeros(torch.cat((R_cube_big[:, 0], torch.
        reshape(J[:, 0, :], (-1, 3, 1))), dim=2)))
    for i in range(1, self.kintree_table.shape[1]):
        results.append(torch.matmul(results[parent[i]], self.with_zeros(
            torch.cat((R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:,
            parent[i], :], (-1, 3, 1))), dim=2))))
    stacked = torch.stack(results, dim=1)
    results = stacked - self.pack(torch.matmul(stacked, torch.reshape(torch
        .cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float64).to(
        self.device)), dim=2), (batch_num, 24, 4, 1))))
    T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 
        3, 1, 2)
    rest_shape_h = torch.cat((v_posed, torch.ones((batch_num, v_posed.shape
        [1], 1), dtype=torch.float64).to(self.device)), dim=2)
    v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
    v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
    result = v + torch.reshape(trans, (batch_num, 1, 3))
    joints = torch.tensordot(result, self.joint_regressor.transpose(1, 0),
        dims=([1], [0])).transpose(1, 2)
    return result
