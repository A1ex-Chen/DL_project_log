def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print
                e
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().
        keys()))))
