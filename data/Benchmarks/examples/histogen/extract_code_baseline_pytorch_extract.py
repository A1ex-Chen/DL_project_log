def extract(lmdb_env, loader, model, device):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader, file=sys.stdout)
        for img, _, filename in pbar:
            img = img.to(device)
            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()
            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')
        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
