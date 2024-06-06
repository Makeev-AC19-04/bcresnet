# Broadcasted Residual Learning for Efficient Keyword Spotting.

This repository is a fork of a repo by [Tijmen Blankevoort](https://github.com/TiRune) and [Iheujo](https://github.com/lheujo).
## Getting started
### Prerequisites
This code requires the following:
- python >= 3.6
- pytorch >= 1.7.1
### Installation
```
conda create -n bcresnet python=3.6
conda activate bcresnet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
conda install tqdm, requests
```

## Usage
### Dataset
Firstly, you have to make directory of a dataset with .vaw files inside `./data` folder. Your `./data` directory should look like this:
```
. └── Data/ 
	└── {KEYWORD}/ 
		├── {KEYWORD} 
		├── filler 
		└── _background_noise_
```
You can also have multiple datasets with different keywords like this:
```
. └── Data/ 
├── {KEYWORD1}/ 
│ ├── {KEYWORD1} 
│ ├── filler 
│ └── _background_noise_ 
├── {KEYWORD2}/ 
│ ├── {KEYWORD2} 
│ ├── filler 
│ └── _background_noise_ 
├── ... 
└── {KEYWORD_N}/ 
	├── {KEYWORD_N} 
	├── filler 
	└── _background_noise_
```
As an example you can download [this dataset.](https://drive.google.com/file/d/1KH62kZaVzsikGz8_Ve9JRe8hO_H0rRI3/view?usp=drive_link)
### Constants
In `constants.py` you can set different parameters:
- `KEYWORD` - keyword you want do work with. **Important:** value of `KEYWORD` must be the same as the dataset directory name. For example, if you set `KEYWORD = "bed"`, your `./data` directory should look like this:
```
. └── Data/ 
	└── bed/ 
		├── bed 
		├── filler 
		└── _background_noise_
```
- `TOTAL_EPOCH` - number of epochs to train model;
- `WARMUP_EPOCH` - number of warmup epochs;
- `TRAIN_RATIO` - percentage of files to use for model training;
- `VAL_RATIO` - percentage of files to use for model validating, `TEST_RATIO` will be calculated as `1 - TRAIN_RATIO - VAL_RATIO`
### Running code

1. To use BCResNet-8 with GPU 0 and split your dataset, run the following command:

```
python main.py --tau 8 --gpu 0 --split_dataset
```

2. To use BCResNet-1 with GPU 1 and skip splitting dataset (for example, it was done with previous command), run the following command:

```
python main.py --tau 1 --gpu 1
```