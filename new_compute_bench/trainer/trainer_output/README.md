---
library_name: transformers
language:
- en
license: mit
base_model: JeremiahZ/roberta-base-stsb
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: trainer_output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# trainer_output

This model is a fine-tuned version of [JeremiahZ/roberta-base-stsb](https://huggingface.co/JeremiahZ/roberta-base-stsb) on the GLUE STSB dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.52.0.dev0
- Pytorch 2.6.0+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1
