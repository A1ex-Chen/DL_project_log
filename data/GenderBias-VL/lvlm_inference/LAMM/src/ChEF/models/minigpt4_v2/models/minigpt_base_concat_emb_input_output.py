def concat_emb_input_output(self, input_embs, input_atts, output_embs,
    output_atts):
    """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
    input_lens = []
    cat_embs = []
    cat_atts = []
    for i in range(input_embs.size(0)):
        input_len = input_atts[i].sum()
        input_lens.append(input_len)
        cat_embs.append(torch.cat([input_embs[i][:input_len], output_embs[i
            ], input_embs[i][input_len:]]))
        cat_atts.append(torch.cat([input_atts[i][:input_len], output_atts[i
            ], input_atts[i][input_len:]]))
    cat_embs = torch.stack(cat_embs)
    cat_atts = torch.stack(cat_atts)
    return cat_embs, cat_atts, input_lens
