# Installation

## TensorFlow

### Docker

#### Installation

```bash
docker pull tensorflow/tensorflow 
docker pull tensorflow/tensorflow:devel-gpu   
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

#### Execution

- CMD

```bash
docker run -it --rm tensorflow/tensorflow python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

- bash

```bash
docker run -it tensorflow/tensorflow bash
```

- script

```bash
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py
```

- Jupyter Notebook

```bash
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-jupyter
```







