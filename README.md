## Modification Overview
This repository is based on the [Pointcept framework](https://github.com/Pointcept/Pointcept) at commit `5b10bc0`. All SIP-specific datasets, configurations, and engines are
implemented on top of that version.

- **`pointcept/datasets/`** 
  - `sip.py` *(added)* - Training and test datalaoders for the [SIP dataset](https://github.com/syoi92/SIP_dataset/).
  - `transform.py` - added **SceneFragmentation** and **manifoldSampling**.

- **`pointcept/engines/`**
  - `train.py`, `test.py` - added **SIPFragmentTrainer** to support fragment-based training workflows
  - `hooks/evaluator.py `, `hooks/misc.py` - added Evaluator and Logger for fragment-based segmentation.
  - `point_transformer/point_transformer_seg.py` - added **Seg74** backbone.

## Usage

### Data preparation

The raw scans are on [Zenodo](https://doi.org/10.5281/zenodo.17667735). Run the preprocessing pipeline from the [SIP_dataset](https://github.com/syoi92/SIP_dataset/) repo (e.g. `preprocessing.py`) so you have **one** training-ready tree; then point this project at it with a symlink under `./data`:

```sh
ln -s /path/to/preprocessed/sip-root data/sip
```

### Training / Test

```sh
sh scripts/train.sh -g "$GPU" -d "$Dataset" -c "$cfg" -n "$name"
sh scripts/test.sh -g "$GPU" -d "$Dataset" -c "$cfg" -n "$name" -w last
```

Example:

```sh
sh scripts/train.sh -g 0 -d sip -c seg-manifold -n seg-manifold-mdl1
```
