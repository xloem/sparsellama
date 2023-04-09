# SparseLLaMA

This repository contains code to reproduce the key results of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774) with the LLaMA model.

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`datautils.py`, `opt.py`, `bloom.py`, `llama.py`) 
* Perform unstructured, n:m and sparse + quantized SparseGPT compression on OPT and BLOOM models. (`sparsegpt.py`, `opt.py`, `bloom.py`, `llama.py`)

Note that this SparseGPT implementation was originally based on the open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## TODO

The output is not yet what a new user might expect:

- [ ] Although weights are set to zero in the passed proportion, at the end they are written out as full dense matrices, requiring the same storage and runtime operations as before unless further change is made.
- [ ] Although weights can be set to quantized values, they are still written to disk in float16. This can likely be improved by pulling a little more code from the GPTQ project.


## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.28.0
* `datasets`: tested on v2.10.1

## Usage

Here is a simple command to sparsify a LLaMA model.
See also the CMD-argument documentation.

```
python llama.py path/to/llama-hf/7B c4 --sparsity 0.5 --blocksize 128 --save llama-7B-sparse-50pct-128blksz-4bit
```

To run on other LLaMA models, replace "path/to/llama-hf/7B" by the path or HuggingFace name of the corresponding model.

The original OPT and BLOOM scripts are also present here and have a very similar interface.

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
