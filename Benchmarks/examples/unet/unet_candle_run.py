def run(gParameters):
    x_train, y_train = unet.load_data()
    model = unet.build_model(420, 580, gParameters['activation'],
        gParameters['kernel_initializer'])
    model.summary()
    model.compile(optimizer=gParameters['optimizer'], loss=
        'binary_crossentropy', metrics=['accuracy'])
    model_chkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1,
        save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=gParameters[
        'batch_size'], epochs=gParameters['epochs'], verbose=1,
        validation_split=0.3, shuffle=True, callbacks=[model_chkpoint])
    return history
