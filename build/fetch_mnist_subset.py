"""
Fetch a small, balanced MNIST subset and pack it into an offline sprite.

Run once:  python build/fetch_mnist_subset.py

Outputs:
  data/mnist_subset.png   grayscale sprite grid of 28x28 digits (row major)
  data/mnist_subset.json  labels + layout metadata

The app loads these locally so nothing depends on the network at class time.
No long dashes are used anywhere in this project.
"""

import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
PER_CLASS_TRAIN = 150   # 150 x 10 = 1500 training digits
PER_CLASS_TEST = 20     # 20 x 10  = 200 held out digits
COLS = 50               # sprite grid width in cells

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))


def load_mnist():
    print("Downloading MNIST from the Keras mirror ...")
    with urllib.request.urlopen(MNIST_URL, timeout=60) as resp:
        raw = resp.read()
    with np.load(io.BytesIO(raw), allow_pickle=True) as f:
        return f["x_train"], f["y_train"], f["x_test"], f["y_test"]


def take_balanced(images, labels, per_class):
    picked_idx = []
    for digit in range(10):
        idx = np.where(labels == digit)[0][:per_class]
        picked_idx.append(idx)
    picked_idx = np.concatenate(picked_idx)
    # interleave classes so any prefix batch stays roughly balanced
    rng = np.random.default_rng(7)
    rng.shuffle(picked_idx)
    return images[picked_idx], labels[picked_idx]


def build_sprite(images):
    count = images.shape[0]
    size = images.shape[1]  # 28
    rows = (count + COLS - 1) // COLS
    sprite = np.zeros((rows * size, COLS * size), dtype=np.uint8)
    for i in range(count):
        r, c = divmod(i, COLS)
        sprite[r * size:(r + 1) * size, c * size:(c + 1) * size] = images[i]
    return sprite, size, rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist()

    tr_x, tr_y = take_balanced(x_train, y_train, PER_CLASS_TRAIN)
    te_x, te_y = take_balanced(x_test, y_test, PER_CLASS_TEST)

    images = np.concatenate([tr_x, te_x], axis=0)
    labels = np.concatenate([tr_y, te_y], axis=0).astype(int).tolist()
    n_train = tr_x.shape[0]
    n_total = images.shape[0]

    sprite, size, rows = build_sprite(images)
    png_path = os.path.join(DATA_DIR, "mnist_subset.png")
    Image.fromarray(sprite, mode="L").save(png_path, optimize=True)

    meta = {
        "imageSize": int(size),
        "cols": COLS,
        "rows": int(rows),
        "count": int(n_total),
        "labels": labels,
        "splits": {"train": [0, int(n_train)], "test": [int(n_train), int(n_total)]},
        "note": "Grayscale sprite, row major. Pixel values 0..255.",
    }
    json_path = os.path.join(DATA_DIR, "mnist_subset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    kb = os.path.getsize(png_path) / 1024
    print("Wrote %s (%.0f KB, %dx%d)" % (png_path, kb, sprite.shape[1], sprite.shape[0]))
    print("Wrote %s (%d labels, %d train / %d test)"
          % (json_path, n_total, n_train, n_total - n_train))


if __name__ == "__main__":
    main()
