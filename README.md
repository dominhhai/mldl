# mldl
Learn ML&amp;DL from scratch

## Install
0. Install `pip` and `virtualenv`
```
$ sudo easy_install pip
$ pip install --upgrade virtualenv
```

1. Create Virtualenv
```
$ virtualenv --system-site-packages .venv
```

2. Install dependences
```
$ source .venv/bin/activate
$ easy_install -U pip
$ pip install -r requirements.txt
```

## Usage
1. Begin by activing Virtualenv
```
$ source .venv/bin/activate
```

2. Exit by deactiving Virtualenv
```
$ deactivate
```

## Folders
* `dataset`: dataset for leanring
* `code`: source code.

> `.tf.py` is code for `TensorFlow`.
>
> Others for normal python program.


## Examples
1. [Linear Regression](/code/linear_regression)

2. [Linear Classification](/code/linear_classification)
