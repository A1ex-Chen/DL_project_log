def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--model_name_or_path', type=str, default=
        'bert-large-uncased')
    parser.add_argument('--learning_rate', type=str, default=5e-05)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=False)
    args, _ = parser.parse_known_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.getLevelName('INFO'), handlers=[
        logging.StreamHandler(sys.stdout)], format=
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'],
        truncation=True, padding='longest'), batched=True)
    train_dataset.set_format(type='tensorflow', columns=['input_ids',
        'attention_mask', 'label'])
    train_features = {x: train_dataset[x] for x in ['input_ids',
        'attention_mask']}
    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features,
        train_dataset['label'])).batch(args.train_batch_size)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'],
        truncation=True, padding='max_length'), batched=True)
    test_dataset.set_format(type='tensorflow', columns=['input_ids',
        'attention_mask', 'label'])
    test_features = {x: test_dataset[x] for x in ['input_ids',
        'attention_mask']}
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_features,
        test_dataset['label'])).batch(args.eval_batch_size)
    learning_rate = args.learning_rate * dist.size()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = dist.DistributedOptimizer(optimizer)
    model = TFAutoModelForSequenceClassification.from_pretrained(args.
        model_name_or_path)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
        experimental_run_tf_function=False)
    BroadcastGlobalVariablesCallback = (dist.callbacks.
        BroadcastGlobalVariablesCallback)
    callbacks = [BroadcastGlobalVariablesCallback(0)]
    if args.do_train:
        logger.info('*** Train ***')
        start_time = time.time()
        train_results = model.fit(tf_train_dataset, epochs=1,
            steps_per_epoch=100, callbacks=callbacks, validation_batch_size
            =args.eval_batch_size, batch_size=args.train_batch_size,
            verbose=1 if dist.rank() == 0 else 0)
        train_runtime = {f'train_runtime': round(time.time() - start_time, 4)}
        logger.info(f'train_runtime = {train_runtime}\n')
    if args.do_eval:
        logger.info('*** Evaluate ***')
        result = model.evaluate(tf_test_dataset, batch_size=args.
            eval_batch_size, return_dict=True)
