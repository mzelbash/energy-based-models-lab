// The book's Python (Keras) listings with expanded, plain language annotations, plus a
// small single pass Python highlighter. Every concept tab shows one of these next to the
// live TensorFlow.js version so students see the real code explained line by line.
// No long dashes are used anywhere in this project.

const KEYWORDS = new Set([
  'def', 'class', 'return', 'for', 'in', 'while', 'with', 'as', 'import', 'from',
  'if', 'elif', 'else', 'and', 'or', 'not', 'is', 'lambda', 'yield', 'pass',
  'True', 'False', 'None', 'self', 'super',
]);
const BUILTINS = new Set(['range', 'len', 'print', 'zip', 'tf', 'np', 'layers', 'models',
  'optimizers', 'activations', 'metrics', 'datasets', 'random']);

function esc(ch) {
  return ch === '&' ? '&amp;' : ch === '<' ? '&lt;' : ch === '>' ? '&gt;' : ch;
}

// Single pass tokenizer, so inserted markup never gets re-matched.
export function highlightPython(src) {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === '#') {
      let j = i;
      while (j < n && src[j] !== '\n') j++;
      out += '<span class="tok-com">' + [...src.slice(i, j)].map(esc).join('') + '</span>';
      i = j;
      continue;
    }
    if (c === '"' || c === "'") {
      const q = c;
      let j = i + 1;
      while (j < n && src[j] !== q && src[j] !== '\n') {
        if (src[j] === '\\') j++;
        j++;
      }
      j = Math.min(j + 1, n);
      out += '<span class="tok-str">' + [...src.slice(i, j)].map(esc).join('') + '</span>';
      i = j;
      continue;
    }
    if (/[0-9]/.test(c) && !/[A-Za-z_]/.test(src[i - 1] || '')) {
      let j = i;
      while (j < n && /[0-9._eE]/.test(src[j])) j++;
      out += '<span class="tok-num">' + src.slice(i, j) + '</span>';
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      const after = src[j];
      if (KEYWORDS.has(word)) out += '<span class="tok-kw">' + word + '</span>';
      else if (after === '(') out += '<span class="tok-fn">' + word + '</span>';
      else if (BUILTINS.has(word)) out += '<span class="tok-cls">' + word + '</span>';
      else out += word;
      i = j;
      continue;
    }
    out += esc(c);
    i++;
  }
  return out;
}

// Render one listing (title, caption, code, annotations) into a DOM element.
export function renderListing(listing) {
  const wrap = document.createElement('div');
  const cap = document.createElement('div');
  cap.className = 'small muted';
  cap.style.margin = '2px 0 6px';
  cap.innerHTML = `<b>${listing.title}</b>${listing.caption ? ' &middot; ' + listing.caption : ''}`;
  wrap.appendChild(cap);

  const pre = document.createElement('pre');
  pre.className = 'code';
  pre.innerHTML = highlightPython(listing.code);
  wrap.appendChild(pre);

  if (listing.annotations && listing.annotations.length) {
    for (const a of listing.annotations) {
      const row = document.createElement('div');
      row.className = 'annot';
      row.innerHTML = `<span class="badge">${a.n}</span><p>${a.text}</p>`;
      wrap.appendChild(row);
    }
  }
  return wrap;
}

// ---------------------------------------------------------------------------
// The listings. Code is the book's, annotations are expanded for teaching.
// ---------------------------------------------------------------------------
export const LISTINGS = {
  load: {
    title: 'Example 7-1',
    caption: 'Loading the MNIST dataset',
    code: `from tensorflow.keras import datasets
(x_train, _), (x_test, _) = datasets.mnist.load_data()`,
    annotations: [
      { n: '1', text: 'We only need the images, not the digit labels, because an EBM is unsupervised. The underscores throw the labels away.' },
    ],
  },
  preprocess: {
    title: 'Example 7-2',
    caption: 'Preprocessing the MNIST dataset',
    code: `def preprocess(imgs):
    imgs = (imgs.astype("float32") - 127.5) / 127.5
    imgs = np.pad(imgs, ((0,0), (2,2), (2,2)), constant_values=-1.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)`,
    annotations: [
      { n: '1', text: 'Pixels start at 0 to 255. Subtracting 127.5 and dividing by 127.5 rescales them to the range -1 to 1, which keeps the network inputs small and centered.' },
      { n: '2', text: 'The image is padded from 28 by 28 to 32 by 32 with the background value of -1. A power of two size divides cleanly through four stride 2 convolutions.' },
      { n: '3', text: 'A trailing channel dimension is added, so each image becomes 32 by 32 by 1 (one grayscale channel).' },
    ],
  },
  energy: {
    title: 'Example 7-3',
    caption: 'Building the energy function E(x)',
    code: `ebm_input = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(16, kernel_size=5, strides=2, padding="same", activation=activations.swish)(ebm_input)
x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation=activations.swish)(x)
ebm_output = layers.Dense(1)(x)
model = models.Model(ebm_input, ebm_output)`,
    annotations: [
      { n: '1', text: 'A stack of Conv2D layers. Each stride of 2 halves the width and height while the channel count grows, so the network reads larger and larger patterns.' },
      { n: '2', text: 'Swish, defined as x times sigmoid(x), is a smooth alternative to ReLU. Smoothness matters here because we later take gradients of this network with respect to its input.' },
      { n: '3', text: 'The final layer is a single unit with linear activation, so the output is one real number that can be any value. That number is the score, and the energy is its negative.' },
    ],
  },
  langevin: {
    title: 'Example 7-4',
    caption: 'The Langevin sampling function',
    code: `def generate_samples(model, inp_imgs, steps, step_size, noise):
    for _ in range(steps):
        inp_imgs += tf.random.normal(inp_imgs.shape, mean=0, stddev=noise)
        inp_imgs = tf.clip_by_value(inp_imgs, -1.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch(inp_imgs)
            out_score = -model(inp_imgs)
        grads = tape.gradient(out_score, inp_imgs)
        grads = tf.clip_by_value(grads, -0.03, 0.03)
        inp_imgs += -step_size * grads
        inp_imgs = tf.clip_by_value(inp_imgs, -1.0, 1.0)
    return inp_imgs`,
    annotations: [
      { n: '1', text: 'Loop for a chosen number of steps. Each pass nudges the image a little closer to something the model scores as realistic.' },
      { n: '2', text: 'A small amount of random noise is added first. Without it the sampler can get stuck in a local dip. This is the stochastic part of stochastic gradient Langevin dynamics.' },
      { n: '3', text: 'Here the weights are fixed. We take the gradient of the output with respect to the INPUT image, which tells us how to change the pixels to lower the energy.' },
      { n: '4', text: 'The gradient is clipped to a small range so no single step can jump too far, then a small multiple of it is added to the image. Repeat, and noise turns into a digit.' },
    ],
  },
  buffer: {
    title: 'Example 7-5',
    caption: 'The replay Buffer',
    code: `class Buffer:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.examples = [
            tf.random.uniform(shape=(1, 32, 32, 1)) * 2 - 1
            for _ in range(128)
        ]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05)
        rand_imgs = tf.random.uniform((n_new, 32, 32, 1)) * 2 - 1
        old_imgs = tf.concat(
            random.choices(self.examples, k=128 - n_new), axis=0
        )
        inp_imgs = tf.concat([rand_imgs, old_imgs], axis=0)
        inp_imgs = generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size, noise=noise
        )
        self.examples = tf.split(inp_imgs, 128, axis=0) + self.examples
        self.examples = self.examples[:8192]
        return inp_imgs`,
    annotations: [
      { n: '1', text: 'The buffer starts as 128 pure noise images. Over training it fills with the best fake samples produced so far.' },
      { n: '2', text: 'About 5 percent of each batch starts fresh from noise, the rest are reused from the buffer, so sampling does not always restart from scratch.' },
      { n: '3', text: 'The mixed batch is run through the Langevin sampler, then the results are pushed back into the buffer, which is trimmed to a maximum length.' },
    ],
  },
  train: {
    title: 'Example 7-6',
    caption: 'EBM trained using contrastive divergence (train_step)',
    code: `def train_step(self, real_imgs):
    real_imgs += tf.random.normal(shape=tf.shape(real_imgs), mean=0, stddev=0.005)
    real_imgs = tf.clip_by_value(real_imgs, -1.0, 1.0)
    fake_imgs = self.buffer.sample_new_exmps(steps=60, step_size=10, noise=0.005)
    inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
    with tf.GradientTape() as training_tape:
        real_out, fake_out = tf.split(self.model(inp_imgs), 2, axis=0)
        cdiv_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        reg_loss = self.alpha * tf.reduce_mean(real_out ** 2 + fake_out ** 2)
        loss = reg_loss + cdiv_loss
    grads = training_tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))`,
    annotations: [
      { n: '1', text: 'A little noise is added to the real images so the model does not simply memorize the training set.' },
      { n: '2', text: 'Fake images are drawn from the buffer, which runs the Langevin sampler for 60 steps to produce believable low energy samples.' },
      { n: '3', text: 'Real and fake images go through the model in one pass, then the outputs are split back apart.' },
      { n: '4', text: 'The contrastive divergence loss is simply mean fake score minus mean real score. Minimizing it pushes real scores up and fake scores down, with no need to normalize.' },
      { n: '5', text: 'A small regularizer keeps the raw scores from drifting to very large magnitudes, which stabilizes training.' },
    ],
  },
  generate: {
    title: 'Example 7-7',
    caption: 'Generating new observations using the EBM',
    code: `start_imgs = np.random.uniform(size=(10, 32, 32, 1)) * 2 - 1
gen_img = generate_samples(
    ebm.model,
    start_imgs,
    steps=1000,
    step_size=10,
    noise=0.005,
    return_img_per_step=True,
)`,
    annotations: [
      { n: '1', text: 'Start from 10 fresh noise images.' },
      { n: '2', text: 'Run the same sampler as during training, but for many more steps (1000). More steps means the noise is guided further downhill into a clean, plausible digit.' },
    ],
  },
  boltzmann: {
    title: 'Boltzmann machine energy',
    caption: 'For contrast: an early EBM',
    code: `# Energy of a fully connected Boltzmann machine
# v are visible units, h are hidden units, all binary
# W, L, J are learned weight matrices
E = -0.5 * (v.T @ L @ v + h.T @ J @ h + v.T @ W @ h)

# Trained with contrastive divergence, but sampled with
# Gibbs sampling: alternate updating v given h, then h given v,
# until the chain reaches equilibrium. This mixes slowly.`,
    annotations: [
      { n: '1', text: 'A Boltzmann machine is also an EBM: it defines an energy over a configuration of binary units. The deep EBM in this app replaces this fixed quadratic form with a neural network.' },
      { n: '2', text: 'The key practical difference is sampling. Boltzmann machines use Gibbs sampling, which is slow to mix. Deep EBMs use Langevin dynamics, which follows input gradients and scales to images.' },
    ],
  },
  jem: {
    title: 'Classifier as an EBM',
    caption: 'A discriminative model defines an energy',
    code: `# A normal classifier outputs 10 logits f(x)
logits = classifier(x)                 # shape [batch, 10]

# The usual softmax gives p(y | x)
p_y_given_x = tf.nn.softmax(logits)

# But the SAME logits also define an energy over inputs:
energy = -tf.reduce_logsumexp(logits, axis=-1)   # E(x)

# Low energy means "this looks like some digit I know",
# high energy means "this does not look like any digit".`,
    annotations: [
      { n: '1', text: 'Nothing here is special to generation. This is an ordinary classifier trained with cross entropy.' },
      { n: '2', text: 'The softmax over the logits is the familiar class probability p(y given x).' },
      { n: '3', text: 'The log sum exp of the same logits, negated, is an energy over the input x. This is the insight behind the paper "Your Classifier is Secretly an Energy-Based Model". A purely discriminative network already carries an energy, so it is an EBM in disguise.' },
    ],
  },
};
