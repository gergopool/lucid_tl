# Lucid - Transfer Learning

An analysis on how transfer learning affects a InceptionV1's
behaviour in terms of weights and activations, by
comparing two instances of InceptionV1: one trained on imagenet and one trained further on the CelebA data set.

For visuals, click [here](https://ai.renyi.hu/visualizing-transfer-learning/index.html).

# Setup

## Code and data

The repo contains some large files with git lfs, so don't be worry if 
the cloning takes some time.

```bash
git clone https://github.com/gergopool/lucid_tl.git
cd lucid_tl
make 
```

## Environment

Conda
```bash
conda env create --file environment.yml
```
Pip
```bash
pip install -r requirements.txt
```

# Research

Find demo notebooks under **notebooks/archive/**.

Please work under the **notebooks/** folder - this is ignored by git. If you want to push a notebook and you need permissions, please contact me.