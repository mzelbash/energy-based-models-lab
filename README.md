# Energy-Based Models, a hands on lab

An interactive teaching app for energy-based models, built to accompany chapter 7 of
David Foster's *Generative Deep Learning*. It runs fully in the browser with TensorFlow.js,
no internet needed once set up. Every concept pairs a plain explanation, a live demo, and
the book's actual Python code annotated line by line.

## Run it

The app must be served over http (browsers block local module and data loading over file://).

```
python -m http.server 8000
```

Then open http://localhost:8000/ in a modern browser (Chrome, Edge, or Firefox).

That is all you need to run the app. The two build steps below are only needed if you want
to regenerate the bundled data or retrain the checkpoints.

## Tabs

0. Start here, what an EBM is and why it matters
1. Intuition, the Boltzmann distribution (interactive)
2. The data, MNIST preprocessing (interactive inspector)
3. The energy function E(x), architecture and swish (live scoring)
4. Langevin sampling, noise into a digit (live animated sampler)
5. Training, contrastive divergence (live training from scratch)
6. Generate and analyze, a gallery and a single generation replay
7. Other EBMs, a live restricted Boltzmann machine with Gibbs sampling
8. Classifier as EBM, a normal classifier read as an energy
9. Code appendix, every listing in one place

## Rebuild steps (optional)

These were already run to produce the files under `data/`.

```
python build/fetch_mnist_subset.py       # writes data/mnist_subset.png + .json
python build/train_checkpoints.py         # writes data/checkpoints/{ebm,clf}
```

The checkpoints are trained in Keras (fast on CPU, the book's framework) and exported to a
compact weights format that the browser loads with model.setWeights. The browser model in
`assets/js/model-defs.js` uses the identical architecture and layer order.

## Notes

- No long dashes are used anywhere in this project.
- All heavy compute runs on the TensorFlow.js WebGL backend in your browser.
- The live training tab trains a fresh model from scratch so students see the real algorithm.
  The preloaded checkpoint gives good generations instantly without waiting.
