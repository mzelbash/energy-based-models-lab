"""
Train the two checkpoints the browser app loads instantly:
  - the deep energy based model (contrastive divergence, Langevin sampling)
  - the small MNIST classifier (used by the "classifier is secretly an EBM" tab)

We train in Keras because it is fast on CPU and it is the exact framework the book uses.
Weights are exported into a compact format the browser reads with model.setWeights, so
the tfjs model built in model-defs.js (identical architecture, same layer order) loads
them directly. No external converter is required.

Run:  python build/train_checkpoints.py
Tunables:  EBM_EPOCHS=50 CLF_EPOCHS=30 python build/train_checkpoints.py

No long dashes are used anywhere in this project.
"""

import json
import os
import struct
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, optimizers, activations

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")

EBM_EPOCHS = int(os.environ.get("EBM_EPOCHS", 60))
CLF_EPOCHS = int(os.environ.get("CLF_EPOCHS", 30))
BATCH = int(os.environ.get("BATCH", 64))
CD_STEPS = int(os.environ.get("CD_STEPS", 60))
STEP_SIZE = float(os.environ.get("STEP_SIZE", 10.0))
NOISE = float(os.environ.get("NOISE", 0.005))
ALPHA = 0.1


# ---------------------------------------------------------------------------
# Data: load the exact sprite the browser uses, then preprocess identically.
# ---------------------------------------------------------------------------
def load_subset():
    meta = json.load(open(os.path.join(DATA_DIR, "mnist_subset.json"), encoding="utf-8"))
    sprite = np.asarray(Image.open(os.path.join(DATA_DIR, "mnist_subset.png")).convert("L"))
    S, cols, count = meta["imageSize"], meta["cols"], meta["count"]
    imgs = np.zeros((count, S, S), dtype=np.float32)
    for i in range(count):
        r, c = divmod(i, cols)
        imgs[i] = sprite[r * S:(r + 1) * S, c * S:(c + 1) * S]
    imgs = imgs / 127.5 - 1.0                       # [0,255] -> [-1,1]
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=-1.0)  # 28 -> 32
    imgs = imgs[..., None]
    labels = np.asarray(meta["labels"], dtype=np.int64)
    trS, trE = meta["splits"]["train"]
    teS, teE = meta["splits"]["test"]
    return (imgs[trS:trE], labels[trS:trE], imgs[teS:teE], labels[teS:teE])


# ---------------------------------------------------------------------------
# Models (identical architecture and layer order to assets/js/model-defs.js).
# ---------------------------------------------------------------------------
def build_energy_model():
    inp = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation=activations.swish)(inp)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=activations.swish)(x)
    out = layers.Dense(1, activation="linear")(x)
    return models.Model(inp, out, name="energy_function")


def build_classifier():
    inp = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    logits = layers.Dense(10, activation="linear", name="logits")(x)
    return models.Model(inp, logits, name="mnist_classifier")


# ---------------------------------------------------------------------------
# Langevin sampler (Example 7-4), compiled for speed. Fixed step count is baked in.
# ---------------------------------------------------------------------------
def make_langevin(model, steps, step_size, noise):
    @tf.function
    def langevin(inp):
        for _ in range(steps):
            inp = inp + tf.random.normal(tf.shape(inp), 0.0, noise)
            inp = tf.clip_by_value(inp, -1.0, 1.0)
            with tf.GradientTape() as tape:
                tape.watch(inp)
                out = -model(inp, training=False)
            grads = tape.gradient(out, inp)
            grads = tf.clip_by_value(grads, -0.03, 0.03)
            inp = inp - step_size * grads
            inp = tf.clip_by_value(inp, -1.0, 1.0)
        return inp

    return langevin


class Buffer:
    """Replay buffer (Example 7-5)."""

    def __init__(self, langevin, size):
        self.langevin = langevin
        self.size = size
        self.examples = list(np.random.uniform(-1, 1, (size, 32, 32, 1)).astype(np.float32))

    def sample(self):
        n_new = max(1, int(self.size * 0.05))
        n_old = self.size - n_new
        rand = np.random.uniform(-1, 1, (n_new, 32, 32, 1)).astype(np.float32)
        idx = np.random.randint(0, len(self.examples), n_old)
        old = np.stack([self.examples[i] for i in idx], 0)
        inp = np.concatenate([rand, old], 0)
        inp = self.langevin(tf.convert_to_tensor(inp)).numpy()
        self.examples = list(inp) + self.examples
        self.examples = self.examples[:8192]
        return tf.convert_to_tensor(inp)


def train_ebm(x_train):
    model = build_energy_model()
    opt = optimizers.Adam(1e-4)
    langevin = make_langevin(model, CD_STEPS, STEP_SIZE, NOISE)
    buffer = Buffer(langevin, BATCH)
    n = x_train.shape[0]
    steps_per_epoch = n // BATCH

    print(f"Training EBM: {EBM_EPOCHS} epochs, {steps_per_epoch} steps/epoch, cd_steps={CD_STEPS}")
    for epoch in range(EBM_EPOCHS):
        t0 = time.time()
        perm = np.random.permutation(n)
        cdiv_v = real_v = fake_v = 0.0
        for s in range(steps_per_epoch):
            batch = x_train[perm[s * BATCH:(s + 1) * BATCH]]
            real = batch + np.random.normal(0, NOISE, batch.shape).astype(np.float32)
            real = np.clip(real, -1, 1)
            fake = buffer.sample()
            inp = tf.concat([tf.convert_to_tensor(real), fake], axis=0)
            with tf.GradientTape() as tape:
                out = model(inp, training=True)
                real_out, fake_out = tf.split(out, 2, axis=0)
                cdiv = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
                reg = ALPHA * tf.reduce_mean(real_out ** 2 + fake_out ** 2)
                loss = reg + cdiv
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            cdiv_v, real_v, fake_v = float(cdiv), float(tf.reduce_mean(real_out)), float(tf.reduce_mean(fake_out))
        print(f"  epoch {epoch + 1:2d}/{EBM_EPOCHS}  cdiv {cdiv_v:+.3f}  "
              f"real {real_v:+.3f}  fake {fake_v:+.3f}  ({time.time() - t0:.1f}s)")
    return model


def train_classifier(x_train, y_train, x_test, y_test):
    model = build_classifier()
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"Training classifier: {CLF_EPOCHS} epochs")
    model.fit(x_train, y_train, epochs=CLF_EPOCHS, batch_size=64,
              validation_data=(x_test, y_test), verbose=2)
    return model


# ---------------------------------------------------------------------------
# Export weights in tfjs setWeights order: layer by layer, kernel then bias.
# ---------------------------------------------------------------------------
def export_weights(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    weights = model.get_weights()  # matches tfjs getWeights order for identical arch
    shapes = [list(w.shape) for w in weights]
    buf = bytearray()
    for w in weights:
        buf += w.astype("<f4").tobytes()
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(buf)
    with open(os.path.join(out_dir, "weights.json"), "w", encoding="utf-8") as f:
        json.dump({"shapes": shapes, "dtype": "float32", "count": len(weights)}, f)
    print(f"  exported {len(weights)} weight tensors to {out_dir} "
          f"({len(buf) / 1024:.0f} KB)")


def main():
    np.random.seed(7)
    tf.random.set_seed(7)
    x_train, y_train, x_test, y_test = load_subset()
    print("Loaded", x_train.shape[0], "train /", x_test.shape[0], "test digits")

    clf = train_classifier(x_train, y_train, x_test, y_test)
    export_weights(clf, os.path.join(CKPT_DIR, "clf"))

    ebm = train_ebm(x_train)
    export_weights(ebm, os.path.join(CKPT_DIR, "ebm"))
    print("Done.")


if __name__ == "__main__":
    main()
