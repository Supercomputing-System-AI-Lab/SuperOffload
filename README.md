## 📙 About

[![](https://img.shields.io/pypi/l/nnsmith)](https://github.com/ise-uiuc/nnsmith/blob/main/LICENSE)

🌟SuperOffload (PRISM)🌟 is a Superchip-centric offloading system that simultaneously uses Hopper GPU, Grace CPU, and NVLink-C2C interconnect more efficiently.

## 🔥 Quick Start

### Install Required Dependencies

```
pip install transformers pandas numpy ninja hjson msgpack tqdm psutil accelerate future pybind11
```
or
```
pip install -r requirements.txt
```

### Install DeepSpeed
<details><summary>Clone the <b>DeepSpeed</b> repository, switch to the <i>superoffload</i> branch, and install it in editable mode and point <i>DEEPSPEED_DIR</i> to the folder <i>[click]</i></summary>

<div>

```
git clone https://github.com/Supercomputing-System-AI-Lab/DeepSpeed.git
cd DeepSpeed
git checkout superoffload
pip install -e .
export DEEPSPEED_DIR="$PWD"
```

</div>
</details>

### Install NVIDIA Apex
<details><summary>Clone the <b>Apex</b> repository, switch to the <i>23.05-devel</i> branch, and install with CUDA extensions enabled <i>[click]</i></summary>

<div>

```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 23.05-devel
pip install -r requirements.txt
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check --no-build-isolation .
```

</div>
</details>


### Download Megatron-DeepSpeed Repository
<details><summary>Clone the <b>Megatron-DeepSpeed</b> repository and switch to the <i>superoffload</i> branch and point <i>MEGATRON-DEEPSPEED_DIR</i> to the folder <i>[click]</i></summary>

<div>

```
git clone https://github.com/Supercomputing-System-AI-Lab/Megatron-DeepSpeed.git
git checkout superoffload
export MEGATRON_DEEPSPEED_DIR="$PWD"
```

</div>
</details>


### Prepare Training Datasets
For convenience, we provide large binary dataset files that you can use directly; due to their size, they are tracked with Git LFS. 
<details><summary>To download and verify <i>[click]</i></summary>

<div>

```bash
# 1. Install Git LFS
# Ubuntu/Debian:
sudo apt-get install git-lfs
# macOS (Homebrew):
brew install git-lfs

# 2. Initialize Git LFS and pull the data
git lfs install
git lfs pull

# 3. Verify the dataset
ls training_datasets/gpt2_text_document.bin  # should be ~400MB
```

</div>
</details>

**Note**: Feel free to contact us if you encounter any issues setting up the environment or using the provided artifacts.

## ⚡ Throughput (Figure 10 & Figure 11)

one chip
```
bash experiment/throughput/one_chip.sh ${MODE} ${MODEL_SIZE} ${MICRO_BATCH} ${ACTIVATION_CHECKPOINTING}
```

4 chips
```
bash experiment/throughput/four_chip.sh ${MODE} ${MODEL_SIZE} ${MICRO_BATCH} ${ACTIVATION_CHECKPOINTING}
```

16 chips
```
bash experiment/throughput/sixteen_chip.sh ${MODE} ${MODEL_SIZE} ${MICRO_BATCH} ${ACTIVATION_CHECKPOINTING}
```

- `MODE`: Select one of `prism`, `zero_offload`, or `zero_infinity` to specify the offloading strategy.
- `MODEL_SIZE`: Set according to the available options in `experiment/model.json`.
- `MICRO_BATCH`: Specify the micro batch size for training.
- `ACTIVATION_CHECKPOINTING`: Set to `true` or `false` to enable or disable activation checkpointing.

The throughput results will be displayed directly in the terminal. To save the output to a file for later review, you can append `| tee output.log` to your command.


## ⤴️ Model Size (Figure 13)

one chip
```
bash experiment/model_size/one_chip.sh ${MODE} ${MODEL_SIZE} 
```

4 chips
```
bash experiment/model_size/four_chip.sh ${MODE} ${MODEL_SIZE} 
```

16 chips
```
bash experiment/model_size/sixteen_chip.sh ${MODE} ${MODEL_SIZE}
```

- `MODE`: Select one of `prism`, `zero_offload`, or `zero_infinity` to specify the offloading strategy.
- `MODEL_SIZE`: Set according to the available options in `experiment/model.json`.

The OOM error results will be displayed directly in the terminal. To save the output to a file for later review, you can append `| tee output.log` to your command.

## 📈 Sequence Length (Figure 12)

4 chips
```
bash experiment/model_size/four_chip.sh ${MODE} ${MODEL_SIZE} ${SEQUENCE_LENGTH} ${ACTIVATION_CHECKPOINTING}
```

8 chips
```
bash experiment/model_size/eight_chip.sh ${MODE} ${MODEL_SIZE} ${SEQUENCE_LENGTH} ${ACTIVATION_CHECKPOINTING}
```

- `MODE`: Select one of `prism-ulysses` or `ulysses`.
- `MODEL_SIZE`: Set according to the available options in `experiment/model.json`.
- `SEQUENCE_LENGTH`: Specify the desired sequence length for training.
- `ACTIVATION_CHECKPOINTING`: Set to `true` or `false` to enable or disable activation checkpointing.

The OOM error results will be displayed directly in the terminal. To save the output to a file for later review, you can append `| tee output.log` to your command.

---

## 📜 Citation

<details><summary><b> 📜 PRISM: Unleashing the Power of Large-Scale LLM Training on Superchips </b> <i>[click :: citation]</i></summary>
<div>

```bibtex
@inproceedings{lian2026prism,
  title = {PRISM: Unleashing the Power of Large-Scale LLM Training on Superchips},
  author = {Lian, Xinyu and Tanaka, Masahiro and Ruwase, Olatunji and Zhang, Minjia},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating System},
  year = {2026}
}
```

</div>
</details>

<p align="center">
    <a href="https://www.asplos-conference.org/asplos2026/"><img src="https://img.shields.io/badge/Paper-ASPLOS'26-a55fed.svg"></a>
    <a href="https://github.com/Supercomputing-System-AI-Lab/SuperOffload"><img src="https://img.shields.io/badge/artifact-git-black.svg"></a>
</p>


## 🙏 Acknowledgement

- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
