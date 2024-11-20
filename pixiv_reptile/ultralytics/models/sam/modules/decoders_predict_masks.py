def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.
    Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings:
    torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """
        Predicts masks.

        See 'forward' for more details.
        """
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.
        weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings
        .shape[0], -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape
    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = self.output_upscaling(src)
    hyper_in_list: List[torch.Tensor] = [self.output_hypernetworks_mlps[i](
        mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)]
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
    iou_pred = self.iou_prediction_head(iou_token_out)
    return masks, iou_pred
