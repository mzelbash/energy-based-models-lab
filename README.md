# Energy-Based Models, a hands on lab

An interactive teaching app for energy-based models, built to accompany chapter 7 of
David Foster's *Generative Deep Learning*. It runs fully in the browser with TensorFlow.js,
no internet needed once loaded. Every concept pairs a plain explanation, a live demo, and
the book's actual Python code annotated line by line.

**Live app: https://mzelbash.github.io/energy-based-models-lab**

Created by Mohamed Elbasheer. Free and open source under the MIT License.

## What is inside

A student who has never heard of an energy-based model should understand it after using this
app. Six steps walk through the whole idea, each one a single page with the same layout: a big
title, one key idea, a few bullet points next to a picture, a live demo, and a one line
takeaway. The book's code sits in a collapsible section at the bottom of each step.

1. The idea: an energy is a score (low means "looks real", high means "does not")
2. From energy to probability: the Boltzmann distribution (interactive)
3. The energy network E(x): the CNN, the data, and live scoring of real digits against noise
4. Langevin sampling: noise into a digit (live sampler, gallery and replay)
5. Training: contrastive divergence, one step at a time, then live training from scratch
6. Your classifier is secretly an EBM: the softmax denominator read as an energy

An **Extras** tab holds the classic restricted Boltzmann machine (live Gibbs sampling) and
all of the code in one place. Use the Back and Next buttons at the bottom of each step, or the
left and right arrow keys, to move through the lab.

## The models: pretrained checkpoints and in-browser training

The app loads two models when it starts, shown by the "checkpoint loaded" badge. Both were
trained offline in Python with Keras and exported to weight files the browser reads with
`model.setWeights`. Nothing is trained at page load, the weights are simply downloaded.

- **The deep energy-based model** (`data/checkpoints/ebm/`): the Chapter 7 energy network
  (four Conv2D swish layers into two dense layers ending in a single scalar), trained with
  contrastive divergence for 60 epochs on the 1,500 image MNIST subset. It powers the Langevin
  Langevin sampling step so digits appear from noise instantly.
- **An ordinary classifier** (`data/checkpoints/clf/`): a small CNN trained with plain cross
  entropy to about 98 percent accuracy, used only in the Classifier as EBM step.

The exact script that produced both is `build/train_checkpoints.py`.

The **Training step** does something different. When you click Train from scratch it builds a
brand new, randomly initialized copy of the same energy-model architecture and trains it live
in your browser using TensorFlow.js, running the real contrastive divergence algorithm and the
Langevin replay buffer from `assets/js/ebm-core.js`. When that run finishes, the app switches
to using your freshly trained model; reload the page to return to the pretrained checkpoint.
The classifier is never affected by in-browser training.

Both the Python pretraining script and the in-browser TensorFlow.js training code are shown in
full inside the app, on the Extras tab.

## Run it locally

You need Python 3 (any recent version) to serve the files. The app must be served over http,
because browsers block local module and data loading over the file:// protocol.

```
git clone https://github.com/mzelbash/energy-based-models-lab.git
cd energy-based-models-lab
python -m http.server 8000
```

Then open http://localhost:8000/ in a modern browser (Chrome, Edge, or Firefox). That is all
you need. No build step, no npm install, nothing to compile.

If you prefer Node instead of Python for the static server:

```
npx serve .
```

## Deploy your own copy on GitHub Pages

1. Fork or clone this repository into your own GitHub account.
2. In your repository, open Settings, then Pages.
3. Under Build and deployment, set Source to Deploy from a branch.
4. Choose the main branch and the / (root) folder, then Save.
5. After about a minute your app is live at https://YOUR-USERNAME.github.io/YOUR-REPO/

The app uses only relative paths and a locally vendored copy of TensorFlow.js, so it works on
a Pages subpath with no changes. A `.nojekyll` file is included so Pages serves the asset and
data folders as they are.

## Rebuild steps (optional)

The bundled data and model checkpoints are already committed, so you do not need these to run
the app. They are here for full reproducibility.

```
python build/fetch_mnist_subset.py     # writes data/mnist_subset.png and .json
python build/train_checkpoints.py       # writes data/checkpoints/{ebm,clf}
python build/verify_ebm.py              # renders generated digits to scratch_out/
```

The checkpoints are trained in Keras (fast on CPU, the book's framework) and exported to a
compact weights format that the browser loads with model.setWeights. The browser model in
`assets/js/model-defs.js` uses the identical architecture and layer order, so the weights load
directly. There is no dependency on a TensorFlow.js converter.

Optional in browser verification uses puppeteer-core against an installed Chrome or Edge:

```
npm install          # dev dependencies for the smoke tests only
python -m http.server 8000    # in one terminal
npm run test:smoke            # in another terminal
```

## Project layout

```
index.html                 The single page app, tabbed layout
assets/css/theme.css       Enterprise theme, three selectable themes
assets/js/                 App modules (data, model, sampler, training, charts, code, ...)
assets/js/vendor/          TensorFlow.js, vendored for offline use
data/                      MNIST sprite subset and trained checkpoints
build/                     Python data and training scripts, plus smoke tests
```

## References and further reading

If you use or build on this app, a citation to the original book is appreciated.

- David Foster. *Generative Deep Learning*, 2nd edition. O'Reilly Media, 2023. Chapter 7,
  Energy-Based Models. This app is a hands on companion to that chapter.
- Yilun Du and Igor Mordatch. Implicit Generation and Modeling with Energy-Based Models. 2019.
  https://arxiv.org/abs/1903.08689
- Geoffrey E. Hinton. Training Products of Experts by Minimizing Contrastive Divergence. 2002.
  https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
- Max Welling and Yee Whye Teh. Bayesian Learning via Stochastic Gradient Langevin Dynamics. 2011.
  https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
- Prajit Ramachandran et al. Searching for Activation Functions (swish). 2017.
  https://arxiv.org/abs/1710.05941
- Will Grathwohl et al. Your Classifier is Secretly an Energy-Based Model and You Should Treat
  it Like One. ICLR 2020. https://arxiv.org/abs/1912.03263
- Phillip Lippe. Tutorial on deep energy-based generative models, which the book's code adapts.

## License

MIT License. See the LICENSE file. You are free to use, modify, and share this app, including
for teaching. The underlying book text and figures remain the property of their publisher and
are not included in this repository.
