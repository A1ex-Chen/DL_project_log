import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.metrics import binary_crossentropy, mean_squared_error

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

DATA_URL = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Examples/rnagen/"

logger = logging.getLogger(__name__)
























class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl")

    def call(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        if type(x) == tuple:  # cvae
            y_pred = self.decoder([z, x[1]])
        else:
            y_pred = self.decoder(z)
        return y_pred

    def custom_step(self, data, train=False):
        with tf.GradientTape() as tape:
            x, y = data
            z_mean, z_log_var, z = self.encoder(x)
            if type(x) == tuple:  # cvae
                y_pred = self.decoder([z, x[1]])
            else:
                y_pred = self.decoder(z)
            reconstruction_loss = keras.losses.binary_crossentropy(y, y_pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        if train:
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {
            "loss": self.total_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
        }
        for m in self.metrics:
            results[m.name] = m.result()
        return results

    def train_step(self, data):
        return self.custom_step(data, train=True)

    def test_step(self, data):
        return self.custom_step(data, train=False)








class VAE(keras.Model):





        usecols = [0] + [
            i for i, c in enumerate(df_cols.columns) if with_prefix(c) in feature_subset
        ]
        df_cols = df_cols.iloc[:, usecols]

    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_csv(path, engine="c", sep="\t", usecols=usecols, dtype=dtype_dict)
    if "Cancer_type_id" in df.columns:
        df.drop("Cancer_type_id", axis=1, inplace=True)

    prefixes = df["Sample"].str.extract("^([^.]*)", expand=False).rename("Source")
    sources = prefixes.drop_duplicates().reset_index(drop=True)
    df_source = pd.get_dummies(sources, prefix="rnaseq.source", prefix_sep=".")
    df_source = pd.concat([sources, df_source], axis=1)

    df1 = df["Sample"]
    if embed_feature_source:
        df_sample_source = pd.concat([df1, prefixes], axis=1)
        df1 = df_sample_source.merge(df_source, on="Source", how="left").drop(
            "Source", axis=1
        )
        logger.info(
            "Embedding RNAseq data source into features: %d additional columns",
            df1.shape[1] - 1,
        )

    df2 = df.drop("Sample", axis=1)
    if add_prefix:
        df2 = df2.add_prefix("rnaseq.")

    if scaling:
        df2 = impute_and_scale(df2, scaling, imputing)

    df = pd.concat([df1, df2], axis=1)

    # scaling needs to be done before subsampling
    if sample_set:
        chosen = df["Sample"].str.startswith(sample_set)
        df = df[chosen].reset_index(drop=True)

    if index_by_sample:
        df = df.set_index("Sample")

    logger.info("Loaded combined RNAseq data: %s", df.shape)

    return df


def load_top_cell_types(n=10, scaling="minmax"):
    df_cell = load_cell_rnaseq(
        use_landmark_genes=True, preprocess_rnaseq="source_scale", scaling=scaling
    )
    df_type = load_cell_type()
    df = pd.merge(df_type, df_cell, on="Sample")
    type_counts = dict(df.type.value_counts().nlargest(n))
    df_top_types = df[df.type.isin(type_counts.keys())]
    return df_top_types


def train_type_classifier(x, y, batch_size=256, epochs=2, verbose=1):
    input_shape = (x.shape[1],)
    num_classes = y.shape[1]

    model = keras.Sequential()
    model.add(layers.Dense(200, input_shape=input_shape, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=verbose,
    )

    return model


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl")

    def call(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        if type(x) == tuple:  # cvae
            y_pred = self.decoder([z, x[1]])
        else:
            y_pred = self.decoder(z)
        return y_pred

    def custom_step(self, data, train=False):
        with tf.GradientTape() as tape:
            x, y = data
            z_mean, z_log_var, z = self.encoder(x)
            if type(x) == tuple:  # cvae
                y_pred = self.decoder([z, x[1]])
            else:
                y_pred = self.decoder(z)
            reconstruction_loss = keras.losses.binary_crossentropy(y, y_pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        if train:
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {
            "loss": self.total_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
        }
        for m in self.metrics:
            results[m.name] = m.result()
        return results

    def train_step(self, data):
        return self.custom_step(data, train=True)

    def test_step(self, data):
        return self.custom_step(data, train=False)


def train_autoencoder(x, y, params={}):
    model_type = params.get("model", "cvae")  # vae or cvae (conditional VAE)
    latent_dim = params.get("latent_dim", 10)

    input_dim = x.shape[1]
    num_classes = y.shape[1]

    # Encoder Part
    x_input = keras.Input(shape=(input_dim,))
    c_input = keras.Input(shape=(num_classes,))

    encoder_input = [x_input, c_input] if model_type == "cvae" else x_input
    h = layers.concatenate(encoder_input) if model_type == "cvae" else x_input
    h = layers.Dense(100, activation="relu")(h)
    h = layers.Dense(100, activation="relu")(h)

    if model_type == "ae":
        encoded = layers.Dense(latent_dim, activation="relu")(h)
    else:
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
        z = Sampling()([z_mean, z_log_var])
        encoded = [z_mean, z_log_var, z]
        # if model_type == 'cvae':
        #     z_cond = keras.layers.concatenate([z, c_input])

    # Decoder Part
    latent_input = keras.Input(shape=(latent_dim,))

    decoder_input = [latent_input, c_input] if model_type == "cvae" else latent_input
    h = layers.concatenate(decoder_input) if model_type == "cvae" else latent_input
    h = layers.Dense(100, activation="relu")(h)
    h = layers.Dense(100, activation="relu")(h)

    decoded = layers.Dense(input_dim, activation="sigmoid")(h)

    # Build autoencoder model
    encoder = keras.Model(encoder_input, encoded, name="encoder")
    decoder = keras.Model(decoder_input, decoded, name="decoder")

    if model_type == "ae":
        model = keras.Model(encoder_input, decoder(encoded))
        metrics = [xent, corr]
        loss = mse
    else:
        model = VAE(encoder, decoder)
        metrics = [xent, mse, corr]
        loss = None

    # encoder.summary()
    # decoder.summary()
    # # model.summary()

    inputs = [x, y] if model_type == "cvae" else x
    outputs = x

    batch_size = params.get("batch_size", 256)
    epochs = params.get("epochs", 100)

    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    model.fit(
        inputs, outputs, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    return model


def xy_from_df(df, shuffle=False):
    if shuffle:
        df = df.sample(frac=1, random_state=0)
    x = df.iloc[:, 2:].values
    y = pd.get_dummies(df.type).values
    return x, y


def main():
    args = parse_args()
    candle.set_seed(args.seed)
    params = vars(args)
    print(f"{args}")

    df = load_top_cell_types(args.top_k_types)
    df_train = df.sample(frac=0.5, random_state=args.seed)
    df_test = df.drop(df_train.index)

    x_train, y_train = xy_from_df(
        df_train, shuffle=True
    )  # shuffle for keras to get random val
    x_test, y_test = xy_from_df(df_test)

    print("\nTrain a type classifier:")
    clf = train_type_classifier(x_train, y_train)

    print("\nEvaluate on test data:")
    results = clf.evaluate(x_test, y_test, batch_size=args.batch_size)

    print("\nTrain conditional autoencoder:")
    model = train_autoencoder(x_train, y_train, params)

    if args.model != "cvae":
        return

    print(f"\nGenerate {args.n_samples} RNAseq samples:")
    start = time.time()
    labels = np.random.randint(0, args.top_k_types - 1, size=args.n_samples)
    c_sample = keras.utils.to_categorical(labels, args.top_k_types)
    z_sample = np.random.normal(size=(args.n_samples, args.latent_dim))
    samples = model.decoder.predict([z_sample, c_sample], batch_size=args.batch_size)
    end = time.time()
    print(
        f"Done in {end-start:.3f} seconds ({args.n_samples/(end-start):.1f} samples/s)."
    )

    print("\nTrain a type classifier with synthetic data:")
    x_new = np.concatenate((x_train, samples), axis=0)
    y_new = np.concatenate((y_train, c_sample), axis=0)
    xy = np.concatenate((x_new, y_new), axis=1)
    np.random.shuffle(xy)
    x_with_syn = xy[:, : x_new.shape[1]]
    y_with_syn = xy[:, x_new.shape[1] :]
    print(f"{x_train.shape[0]} + {args.n_samples} = {x_with_syn.shape[0]} samples")
    clf2 = train_type_classifier(x_with_syn, y_with_syn)

    print("\nEvaluate again on original test data:")
    results2 = clf2.evaluate(x_test, y_test, batch_size=args.batch_size)
    acc, acc2 = results[1], results2[1]
    change = (acc2 - acc) / acc * 100
    print(f"Test accuracy change: {change:+.2f}% ({acc:.4f} -> {acc2:.4f})")

    if not args.plot:
        return

    print("\nPlot test accuracy using models trained with and without synthetic data:")
    print("training time: before vs after")
    rows = []
    for epochs in range(1, 21):
        c1 = train_type_classifier(x_train, y_train, epochs=epochs, verbose=0)
        c2 = train_type_classifier(x_with_syn, y_with_syn, epochs=epochs, verbose=0)
        r1 = c1.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        r2 = c2.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        print(f"# epochs = {epochs}: {r1[1]:.4f} vs {r2[1]:.4f}")
        rows.append(
            {
                "Epochs": epochs,
                "trained w/o synthetic data": r1[1],
                "trained w/ synthetic data": r2[1],
            }
        )
    df = pd.DataFrame(rows).set_index("Epochs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"Type classifier accuray on holdout data ({args.top_k_types} types)"
    plt.figure(dpi=300)
    ax = df.plot(title=title, ax=plt.gca(), xticks=[1, 5, 10, 15, 20])
    ax.set_ylim(0.35, 1)
    prefix = f"test-accuracy-comparison-{args.top_k_types}-types"
    plt.savefig(f"{prefix}.png")
    df.to_csv(f"{prefix}.csv")


if __name__ == "__main__":
    main()