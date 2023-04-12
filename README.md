# JGCF
A simple, efficient and effective Jacobi polynomial-based graph collaborative filtering algorithm.

## Requirements

```
conda env create -f environment.yaml
```

## Quich Start
```
python run.py --dataset gowalla
```

## Datasets

### For Amazon_Books

For alibaba, you can download Amazon_Books.zip from Google Drive



### For Alibaba-iFashion

For alibaba, you can download alibaba.zip from [Google Drive](https://drive.google.com/file/d/1Th7ii_Z0l6AjGq8zWsKuLVCsacIO1AQJ/view?usp=sharing). Then
```
mv alibaba.zip dataset
unzip alibaba.zip
python run.py --dataset alibaba
```

### Empirical Experiments on More Datasets without Sampling

<img decoding="async" src="./assets/ml_100k_spectral_trans.png" width="32%">
<img decoding="async" src="./assets/ml1m_spectral_trans.png" width="31%">
<img decoding="async" src="./assets/pinterest_spectral_trans.png" width="32%">

