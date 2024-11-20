def rename_state_dict(sd, embedding):
    sd = {rename_state_dict_key(k): v for k, v in sd.items()}
    sd['embed_time.emb.weight'] = embedding['weight']
    return sd
