# TripNet
## Abstract
TripNet is a feature extraction network for fashion images.
In training, we use triplet image sets and 3 networks sharing their parameters.
Minimizing the triplet loss, we learn the parameters.
After training, TripNet simply works as a vectorizer.
Here is the implementation with Python+TensorFlow.

## Requirements
Our environment is built on Windows 10 with Anaconda.
TripNet requires below:

* Python 3.5
* TensorFlow 1.1 or 1.3
* OpenCV3 (Get it with `conda install -c menpo opencv3` or `pip install opencv-python`)
* any other packages (Please install them using `pip`)

If the requirements is met, maybe you can also run it on the other platforms.

## Usage
Run the training first:
```
$ python main.py --train --data_list <DATA_LIST_OF_TRIPLET> --data_dir <IMAGES_DIRECTORY>
```
To visualize with TensorBoard's Embedding Projector:
```
$ python main.py --visualize --data_dir <IMAGES_DIRECTORY>
$ tensorboard --logdir ./projector
```
If you want to specify detail, run `python main.py --help` and check the arguments.

## Reference
* Edgar Simo-Serra and Hiroshi Ishikawa, "Fashion Style in 128 Floats: Joint Ranking and Classification using Weak Data for Feature Extraction", *Conference in Computer Vision and Pattern Recognition (CVPR), 2016*
* Sean Bell and Kavita Bala, "Learning Visual Similarity for Product Design with Convolutional Neural Networks", *ACM Trans. on Graphics (SIGGRAPH), 2015*
* Hoffer E. and Ailon N., "Deep metric learning using triplet network.", *International Conference on Learning Representations (ICLR), 2015*
