def getcompelembs(prps):
    embl = []
    for prp in prps:
        embl.append(compel.build_conditioning_tensor(prp))
    return torch.cat(embl)
