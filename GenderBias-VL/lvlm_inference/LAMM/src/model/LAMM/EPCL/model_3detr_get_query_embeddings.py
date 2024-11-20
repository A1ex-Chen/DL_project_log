def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
    query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
    query_inds = query_inds.long()
    query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in
        range(3)]
    query_xyz = torch.stack(query_xyz)
    query_xyz = query_xyz.permute(1, 2, 0)
    pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
    query_embed = self.query_projection(pos_embed)
    return query_xyz, query_embed
