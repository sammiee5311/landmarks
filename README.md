# landmarks
+ display the body landmarks in real time thru webcam.

## How to use
+ download pretrained models(coco, mpi, body25)
+ put the pretrained models
    + pose/coco/
    + pose/mpi
    + pose/body_25
+ run main.py

## Requirements
+ Cmake
+ opencv-python
+ imutils

## classes
### coco
+ nose: 0, neck: 1, right shoulder: 2
+ right elbow: 3, right wrist: 4, left shoulder: 5
+ left elbow: 6, left wrist: 7, right hip: 8
+ right knee: 9, right ankle: 10, left hip: 11
+ left knee: 12, left ankle: 13, right eye: 14
+ left eye: 15, right ear: 16, left ear: 17

### mpi
+ head: 0, neck: 1, right shoulder: 2
+ right elbow: 3, right wrist: 4, left shoulder: 5
+ left elbow: 6, left wrist: 7, right hip: 8
+ right knee: 9, right ankle: 10, left hip: 11
+ left knee: 12, left ankle: 13, chest: 14 
+ background: 15

## Images
<p float="center">
  <img src="https://github.com/sammiee5311/landmarks/blob/master/img/test.gif" width="450" heights="405"/>
</p>

I've got around 23 fps with mpi model which is best result among 3 models.

### Reference
+ https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
