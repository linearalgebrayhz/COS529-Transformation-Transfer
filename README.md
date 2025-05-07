# COS529-Transformation-Transfer

This code base is build upon [PhotoDoodle(2025)](https://arxiv.org/abs/2502.14397)
Please refer to this [repo](https://github.com/showlab/PhotoDoodle) for orginal implementation.

## Quick Start
### 1. **Environment setup**
```bash
git@github.com:linearalgebrayhz/COS529-Transformation-Transfer.git
cd PhotoDoodle

conda create -n doodle python=3.11.10
conda activate doodle
```
### 2. **Requirements installation**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
```
Download the huggingface version packages.

## Dataset

This project uses open-source dataset [BlendedMVS](https://arxiv.org/abs/1911.10127)
You can find the dataset download link in this github [repo](https://github.com/YoYo000/BlendedMVS)

After downloading the data, unzip the file and place the files under `data/` directory.

Two jsonl dataset are already available for use:
`meta.jsonl` PhotoDoodle style dataset containing source view, target view and empty text prompt, which can be used in first stage training.
`transformation_dataset` contains source, target, test source images as well as relative transformations.

You can also generate new data with the provided python program: `blend.py`, `transform_data_converter.py`

Data visualization is available through `visual.py`



