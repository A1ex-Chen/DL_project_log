def run(params):
    x_train, y_train, x_val, y_val = st.load_data(params)
    model = st.transformer_model(params)
    kerasDefaults = candle.keras_default_config()
    optimizer = candle.build_optimizer(params['optimizer'], params[
        'learning_rate'], kerasDefaults)
    model.compile(loss=params['loss'], optimizer=optimizer, metrics=['mae',
        st.r2])
    checkpointer = ModelCheckpoint(filepath=
        'smile_regress.autosave.model.h5', verbose=1, save_weights_only=
        True, save_best_only=True)
    csv_logger = CSVLogger('smile_regress.training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience
        =20, verbose=1, mode='auto', epsilon=0.0001, cooldown=3, min_lr=1e-09)
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1,
        mode='auto')
    history = model.fit(x_train, y_train, batch_size=params['batch_size'],
        epochs=params['epochs'], verbose=1, validation_data=(x_val, y_val),
        callbacks=[checkpointer, csv_logger, reduce_lr, early_stop])
    model.load_weights('smile_regress.autosave.model.h5')
    return history
