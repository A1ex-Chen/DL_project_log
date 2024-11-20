def main():
    args = parse_args()
    candle.set_seed(args.seed)
    params = vars(args)
    print(f'{args}')
    df = load_top_cell_types(args.top_k_types)
    df_train = df.sample(frac=0.5, random_state=args.seed)
    df_test = df.drop(df_train.index)
    x_train, y_train = xy_from_df(df_train, shuffle=True)
    x_test, y_test = xy_from_df(df_test)
    print('\nTrain a type classifier:')
    clf = train_type_classifier(x_train, y_train)
    print('\nEvaluate on test data:')
    results = clf.evaluate(x_test, y_test, batch_size=args.batch_size)
    print('\nTrain conditional autoencoder:')
    model = train_autoencoder(x_train, y_train, params)
    if args.model != 'cvae':
        return
    print(f'\nGenerate {args.n_samples} RNAseq samples:')
    start = time.time()
    labels = np.random.randint(0, args.top_k_types - 1, size=args.n_samples)
    c_sample = keras.utils.to_categorical(labels, args.top_k_types)
    z_sample = np.random.normal(size=(args.n_samples, args.latent_dim))
    samples = model.decoder.predict([z_sample, c_sample], batch_size=args.
        batch_size)
    end = time.time()
    print(
        f'Done in {end - start:.3f} seconds ({args.n_samples / (end - start):.1f} samples/s).'
        )
    print('\nTrain a type classifier with synthetic data:')
    x_new = np.concatenate((x_train, samples), axis=0)
    y_new = np.concatenate((y_train, c_sample), axis=0)
    xy = np.concatenate((x_new, y_new), axis=1)
    np.random.shuffle(xy)
    x_with_syn = xy[:, :x_new.shape[1]]
    y_with_syn = xy[:, x_new.shape[1]:]
    print(
        f'{x_train.shape[0]} + {args.n_samples} = {x_with_syn.shape[0]} samples'
        )
    clf2 = train_type_classifier(x_with_syn, y_with_syn)
    print('\nEvaluate again on original test data:')
    results2 = clf2.evaluate(x_test, y_test, batch_size=args.batch_size)
    acc, acc2 = results[1], results2[1]
    change = (acc2 - acc) / acc * 100
    print(f'Test accuracy change: {change:+.2f}% ({acc:.4f} -> {acc2:.4f})')
    if not args.plot:
        return
    print(
        '\nPlot test accuracy using models trained with and without synthetic data:'
        )
    print('training time: before vs after')
    rows = []
    for epochs in range(1, 21):
        c1 = train_type_classifier(x_train, y_train, epochs=epochs, verbose=0)
        c2 = train_type_classifier(x_with_syn, y_with_syn, epochs=epochs,
            verbose=0)
        r1 = c1.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        r2 = c2.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        print(f'# epochs = {epochs}: {r1[1]:.4f} vs {r2[1]:.4f}')
        rows.append({'Epochs': epochs, 'trained w/o synthetic data': r1[1],
            'trained w/ synthetic data': r2[1]})
    df = pd.DataFrame(rows).set_index('Epochs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = (
        f'Type classifier accuray on holdout data ({args.top_k_types} types)')
    plt.figure(dpi=300)
    ax = df.plot(title=title, ax=plt.gca(), xticks=[1, 5, 10, 15, 20])
    ax.set_ylim(0.35, 1)
    prefix = f'test-accuracy-comparison-{args.top_k_types}-types'
    plt.savefig(f'{prefix}.png')
    df.to_csv(f'{prefix}.csv')
