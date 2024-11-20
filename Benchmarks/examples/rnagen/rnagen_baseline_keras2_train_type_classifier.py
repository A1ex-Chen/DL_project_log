def train_type_classifier(x, y, batch_size=256, epochs=2, verbose=1):
    input_shape = x.shape[1],
    num_classes = y.shape[1]
    model = keras.Sequential()
    model.add(layers.Dense(200, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=
        0.1, verbose=verbose)
    return model
