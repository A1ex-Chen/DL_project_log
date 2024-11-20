def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head,
    pos_mlp, attn_mask=None, padding_mask=None):
    """Perform the forward pass through the entire decoder."""
    output = embed
    dec_bboxes = []
    dec_cls = []
    last_refined_bbox = None
    refer_bbox = refer_bbox.sigmoid()
    for i, layer in enumerate(self.layers):
        output = layer(output, refer_bbox, feats, shapes, padding_mask,
            attn_mask, pos_mlp(refer_bbox))
        bbox = bbox_head[i](output)
        refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
        if self.training:
            dec_cls.append(score_head[i](output))
            if i == 0:
                dec_bboxes.append(refined_bbox)
            else:
                dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(
                    last_refined_bbox)))
        elif i == self.eval_idx:
            dec_cls.append(score_head[i](output))
            dec_bboxes.append(refined_bbox)
            break
        last_refined_bbox = refined_bbox
        refer_bbox = refined_bbox.detach() if self.training else refined_bbox
    return torch.stack(dec_bboxes), torch.stack(dec_cls)
