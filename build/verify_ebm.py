"""
Load the exported EBM checkpoint back into a Keras model and generate digits, to confirm
the checkpoint the browser loads actually produces digit like samples. Fast on TF CPU.
Writes scratch_out/generated.png and scratch_out/real.png.
No long dashes are used anywhere in this project.
"""
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, activations

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CKPT = os.path.join(DATA, "checkpoints", "ebm")
OUT = os.path.normpath(os.path.join(HERE, "..", "scratch_out"))


def build():
    inp = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation=activations.swish)(inp)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=activations.swish)(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out)


def load_weights(model):
    meta = json.load(open(os.path.join(CKPT, "weights.json")))
    raw = np.fromfile(os.path.join(CKPT, "weights.bin"), dtype="<f4")
    ws, off = [], 0
    for shape in meta["shapes"]:
        n = int(np.prod(shape))
        ws.append(raw[off:off + n].reshape(shape))
        off += n
    model.set_weights(ws)


def langevin(model, x, steps, step_size=10.0, noise=0.005):
    x = tf.convert_to_tensor(x)
    for _ in range(steps):
        x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), 0.0, noise), -1, 1)
        with tf.GradientTape() as t:
            t.watch(x)
            out = -model(x, training=False)
        g = tf.clip_by_value(t.gradient(out, x), -0.03, 0.03)
        x = tf.clip_by_value(x - step_size * g, -1, 1)
    return x.numpy()


def save_grid(batch, path, cols=8, scale=4):
    n, h, w = batch.shape[0], 32, 32
    rows = (n + cols - 1) // cols
    out = np.zeros((rows * h * scale, cols * w * scale), dtype=np.uint8)
    for i in range(n):
        img = np.clip((batch[i, :, :, 0] + 1) * 127.5, 0, 255).astype(np.uint8)
        img = np.kron(img, np.ones((scale, scale), dtype=np.uint8))
        r, c = divmod(i, cols)
        out[r * h * scale:(r + 1) * h * scale, c * w * scale:(c + 1) * w * scale] = img
    Image.fromarray(out, "L").save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build()
    load_weights(model)

    start = np.random.uniform(-1, 1, (16, 32, 32, 1)).astype(np.float32)

    def energy(x):
        return float(tf.reduce_mean(-model(x, training=False)))

    print("energy of noise      :", round(energy(start), 3))
    for steps in (200, 600, 1000):
        gen = langevin(model, start.copy(), steps)
        print(f"energy after {steps:4d} steps:", round(energy(gen), 3))
    save_grid(gen, os.path.join(OUT, "generated.png"))
    print("wrote", os.path.join(OUT, "generated.png"))


if __name__ == "__main__":
    main()
