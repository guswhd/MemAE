[TensorFlow 2] Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection
=====
s
TensorFlow implementation of Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection. [<a href="https://github.com/donggong1/memae-anomaly-detection">PyTorch Version</a>] [<a href="https://github.com/YeongHyeon/MemAE">TensorFlow 1 Version</a>]

## Architecture
<div align="center">
  <img src="./figures/memae.png" width="500">  
  <p>Architecture of MemAE.</p>
</div>

## Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="500">  
  <p>Graph of MemAE.</p>
</div>

## Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="600">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by MemAE.</p>
</div>

<div align="center">
  <img src="./figures/test-box.png" width="350"><img src="./figures/histogram-test.png" width="390">
  <p>Box plot and histogram of restoration loss in test procedure.</p>
</div>


# implementaion

#### colab에서 수행

1. **clone repository**
```
!git clone https://github.com/YeongHyeon/MemAE-TF2.git
```

2.  train
```
!git python train.py
```

3. test
```
!git python test.py
```


## Reference
[1] Dong Gong et al. (2019). <a href="https://arxiv.org/abs/1904.02639">Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection</a>. arXiv preprint arXiv:1904.02639.

