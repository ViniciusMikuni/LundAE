# Graph AE with Lund observables

# Requirements

[Tensorflow 1.14](https://www.tensorflow.org/)

[h5py](https://www.h5py.org/)

# Instructions

Download the repo with 

```bash
git clone https://github.com/ViniciusMikuni/LundAE.git
```

The training files are by default assumed to be stored in the folder ```h5```. If a different folder is used, change the ```--data_dir``` flag location of ```train.py```.
To run the training use:
```bash
cd scripts
python train.py [--adj] --log_dir lund_classification
```

The result of each training epoch will be saved uder the folder ```logs/lund_classification```. If the ```--adj``` is called, the script will read the adjancecy matrix from the inputs, otherwise the script calculates the adjancecy matrix, grouping together particles close in the lund-plane.