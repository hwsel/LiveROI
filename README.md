# LiveROI

This is the source code for our LiveROI

## Enviroment:

- OS: Ubuntu 16.4

- Deep Learning Library: [Caffe](https://github.com/BVLC/caffe)

## 3D CNN model

The core part of this project is action recognition based on 3D CNN model. This neural network model is based on Caffe.

- The files:

```

/caffe_3d

/data_list

/doc_files

/models_ECO_Full

/models_ECO_Lite

```

- Install the dependencies with

```
for req in $(cat requirements.txt); do pip install $req; done
```

- Compile 

```
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
make all
make test
make runtest
```
  


## Word Embedding

- Files:

The word analysis is carried out by the following source code:

```
\scripts\online_recognition_LiveROI\phrase2vec.py
```

In the LiveROI project code, this file should be imported.


## LiveROI code:


- Files:

The following file shows the basic demo of how the action recognition works.

```
\scripts\online_recognition_LiveROI\MultiThreadActionRecognition.py
```

The following file shows the LiveROI based on the action recognition and word embedding.

```
\scripts\online_recognition_LiveROI\LiveROI.py
```