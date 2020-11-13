# A Hierarchical Knowledge and Interest Propagation Network for Recommender Systems

This repository is the implementation of paper:

> A Hierarchical Knowledge and Interest Propagation Network for Recommender Systems
>
> Qinghong Chen, Huobin Tan, Guangyan Lin, Ze Wang

## Required packages

The code has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
- torch==1.3.0
- torchvision==0.4.1
- numpy==1.17.3
- pandas==0.25.1
- scikit-learn==0.21.3

## Files in the folder

- data/: The dataset used in the paper. Please download it at https://pan.baidu.com/s/1sw0-S6mEiRYySTZ748ItiQ , the code is 1113.
- `src/`: Implementation of HKIPN.

## Perprocess the data  & run

After downloading the data, you need to preprocess and run by following the steps below:


- Music

```
$ python preprocess.py --dataset music
$ python main.py --dataset music 
```


- Book
```
$ python preprocess.py --dataset book
$ python main.py --dataset book
```

- Movie
```
$ python preprocess.py --dataset movie
$ python main.py --dataset movie
```
