## About GraphscoreDTA

GraphscoreDTA is an optimized graph neural network for protein-ligand binding affinity prediction.  

The benchmark dataset can be found in `./test_set/`. The GraphscoreDTA model is available in `./src/`. And the result will be generated in `./result/`. See our paper for more details.
#### [IMPORTANT]  We provide the input files in the release page (https://github.com/KailiWang1/GraphscoreDTA/releases/tag/Data). Please download it to `./test_set/`.

### Requirements:
- python 3.7.11
- pytorch 1.9.0
- scikit-learn 0.24.2
- dgl 0.9.1.post1
- tqdm 4.62.2
- ipython 7.27.0
- numpy 1.20.3
- pandas 1.3.2
- numba 0.53.1
- scipy 1.7.1

### Installation

In order to get GraphscoreDTA, you need to clone this repo:

```bash
git clone https://github.com/CSUBioGroup/GraphscoreDTA
cd GraphscoreDTA
```
The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment_gpu.yml
conda activate GraphscoreDTA
```
### Predict

to use our model
```bash
cd ./src/
python predict.py
```

### Training

to train your own model
```bash
cd ./src/
python train.py
```

### contact
Kaili Wang: kailiwang@csu.edu.cn
You can also download the codes from https://github.com/KailiWang1/GraphscoreDTA