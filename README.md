## Modification Overview
This repository is based on the [Pointcept framework](https://github.com/Pointcept/Pointcept) at commit `5b10bc0`. All SIP-specific datasets, configurations, and engines are
implemented on top of that version.

- **`pointcept/datasets/sip.py`** *(added)*  
  Training and test datalaoders for the [SIP dataset](https://github.com/syoi92/SIP_dataset/).

- **`pointcept/datasets/transform.py`** *(modified)*  
  Added **SceneSampling** and **SceneFragmentation** transforms.

- **`pointcept/engines/train.py`** *(modified)*  
  Added **`SIPFragmentTrainer`** to support fragment-based training workflows.

- **`pointcept/engines/hooks/evaluator.py`** *(modified)*  
  Added **`SIPSemSegEvaluator`** for fragment-based semantic segmentation evaluation.
