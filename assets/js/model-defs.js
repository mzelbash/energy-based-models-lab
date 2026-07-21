// Shared model definitions used by BOTH the browser app and the Node build scripts.
// Every function takes the TensorFlow.js module as an argument so the exact same
// architecture is used in the browser (global `tf` from the vendored script) and in
// Node (imported `@tensorflow/tfjs`). Keeping one source of truth means a checkpoint
// trained by the build script loads cleanly into the browser model.
//
// No long dashes are used anywhere in this project.

export const IMG_SIZE = 32; // MNIST is padded from 28 to 32, matching the book

// ---------------------------------------------------------------------------
// The energy function E(x): a stack of Conv2D layers with swish activation that
// gradually shrinks the image while growing the channel count, ending in a single
// linear unit so the output is one real number in the range (-inf, +inf).
// This mirrors Example 7-3 in the book.
// ---------------------------------------------------------------------------
export function buildEnergyModel(tf) {
  const input = tf.input({ shape: [IMG_SIZE, IMG_SIZE, 1] });

  let x = tf.layers
    .conv2d({ filters: 16, kernelSize: 5, strides: 2, padding: 'same', activation: 'swish' })
    .apply(input);
  x = tf.layers
    .conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'swish' })
    .apply(x);
  x = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'swish' })
    .apply(x);
  x = tf.layers
    .conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'swish' })
    .apply(x);
  x = tf.layers.flatten().apply(x);
  x = tf.layers.dense({ units: 64, activation: 'swish' }).apply(x);
  const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(x);

  return tf.model({ inputs: input, outputs: output, name: 'energy_function' });
}

// ---------------------------------------------------------------------------
// A small, ordinary MNIST classifier. It is trained with plain cross entropy,
// nothing special. In the "your classifier is secretly an EBM" tab we reinterpret
// its logits as an energy: E(x) = -logsumexp(logits). The point is that a purely
// discriminative model already defines an energy over inputs.
// ---------------------------------------------------------------------------
export function buildClassifier(tf) {
  const input = tf.input({ shape: [IMG_SIZE, IMG_SIZE, 1] });

  let x = tf.layers
    .conv2d({ filters: 16, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu' })
    .apply(input);
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x); // 32 -> 16
  x = tf.layers
    .conv2d({ filters: 32, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu' })
    .apply(x);
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x); // 16 -> 8
  x = tf.layers.flatten().apply(x);
  x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
  const logits = tf.layers.dense({ units: 10, activation: 'linear', name: 'logits' }).apply(x);

  return tf.model({ inputs: input, outputs: logits, name: 'mnist_classifier' });
}
