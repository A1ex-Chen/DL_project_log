def train_epochs(model, train_generator, val_generator, epoch_count,
    learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=
        'categorical_crossentropy', metrics=['accuracy'])
    history_fine = model.fit(train_generator, steps_per_epoch=len(
        train_generator), epochs=epoch_count, validation_data=val_generator,
        validation_steps=len(val_generator), batch_size=BATCH_SIZE)
    return model
