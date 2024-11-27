# Semantic Segmentation

## Solution 1, Use Florence2 and Segment anything
This folder include code aiming to use florence 2 to detect the object, output box, and use Same to output the musk.

### How it works

First, leverage Florence 2 large to detect the box
Second, use the box to output the musk.

And boom, we have the magical semantic object detector.

### What is the result

The box may include multiple items which lead to fail drawing the mask output. 


## Solution 2, Use GroundingDINO
https://github.com/IDEA-Research/GroundingDINO