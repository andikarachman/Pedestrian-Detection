# Pedestrian Detection


## Introduction
We build a pedestrian detection system by by combining Histogram of Oriented Gradients (HoG) feature and support vector machine (SVM). HoG feature provides a reasonable and feature invariant object representation, while SVM framework gives us a robust classifier that can control both the training set error and the classifier's complexity. Complete descriptions of the technique are given the this [paper](https://ieeexplore.ieee.org/document/6835619). 

While it is possible to detect pedestrians from images, videos, and webcam streaming, we only focus on detection from video.


## Dependencies
  * opencv
  * numpy
  * imutils

## Command format

The input of this system will be video, so we need to specify the path to the input and output video.

_$ python pedestrian-detection.py [-h] -i INPUT -o OUTPUT_

- INPUT: path to input video
- OUTPUT: path to output video

