// A tiny restricted Boltzmann machine with contrastive divergence (CD-1) training and
// Gibbs sampling, for the "other energy based models" tab. Visible units are the 1024
// binarized pixels of a 32x32 digit, hidden units are a small binary code. The point is
// to feel how Gibbs sampling alternates between layers and mixes slowly, in contrast to
// Langevin dynamics.
// No long dashes are used anywhere in this project.

const tf = window.tf;
const NV = 32 * 32;
const NH = 64;

export class RBM {
  constructor() {
    this.W = tf.variable(tf.randomNormal([NV, NH], 0, 0.01));
    this.vb = tf.variable(tf.zeros([NV]));
    this.hb = tf.variable(tf.zeros([NH]));
  }

  // Binarize a [-1,1] image batch [n,32,32,1] into {0,1} visible vectors [n,1024].
  static binarize(batch) {
    return tf.tidy(() => batch.reshape([batch.shape[0], NV]).greater(0).toFloat());
  }

  hGivenV(v) {
    return tf.tidy(() => v.matMul(this.W).add(this.hb).sigmoid());
  }
  vGivenH(h) {
    return tf.tidy(() => h.matMul(this.W.transpose()).add(this.vb).sigmoid());
  }
  static sample(p) {
    return tf.tidy(() => tf.randomUniform(p.shape).less(p).toFloat());
  }

  // One CD-1 update on a batch of visible vectors.
  cdStep(v0, lr) {
    tf.tidy(() => {
      const h0p = this.hGivenV(v0);
      const h0 = RBM.sample(h0p);
      const v1p = this.vGivenH(h0);
      const v1 = RBM.sample(v1p);
      const h1p = this.hGivenV(v1);

      const n = v0.shape[0];
      const posW = v0.transpose().matMul(h0p).div(n);
      const negW = v1.transpose().matMul(h1p).div(n);
      this.W.assign(this.W.add(posW.sub(negW).mul(lr)));
      this.vb.assign(this.vb.add(v0.sub(v1).mean(0).mul(lr)));
      this.hb.assign(this.hb.add(h0p.sub(h1p).mean(0).mul(lr)));
    });
  }

  async train(dataBatch, { epochs = 8, batchSize = 64, lr = 0.05, onEpoch = null } = {}) {
    const v = RBM.binarize(dataBatch);
    const n = v.shape[0];
    for (let e = 0; e < epochs; e++) {
      const steps = Math.floor(n / batchSize);
      for (let s = 0; s < steps; s++) {
        const vb = tf.tidy(() => v.slice([s * batchSize, 0], [batchSize, -1]));
        this.cdStep(vb, lr);
        vb.dispose();
      }
      if (onEpoch) onEpoch(e + 1, epochs);
      await tf.nextFrame();
    }
    v.dispose();
  }

  // One full Gibbs step v -> h -> v, returning the visible probabilities as an image.
  gibbsOnce(v) {
    return tf.tidy(() => {
      const h = RBM.sample(this.hGivenV(v));
      const vp = this.vGivenH(h);
      return { vSample: RBM.sample(vp), vProb: vp };
    });
  }

  dispose() {
    this.W.dispose(); this.vb.dispose(); this.hb.dispose();
  }
}

export function imageToVisible(batch1) {
  return RBM.binarize(batch1); // [1,1024]
}

export function visibleToImage(v) {
  return tf.tidy(() => v.reshape([32, 32, 1]).mul(2).sub(1)); // back to [-1,1] for drawing
}
