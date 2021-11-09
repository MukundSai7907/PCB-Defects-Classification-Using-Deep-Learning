# PCB-Defects-Detection-and-Classification

## This is an implementation of the paper: https://arxiv.org/pdf/1901.08204.pdf

## Overview: 
The paper proposes a method to first detect PCB defects using template matching and image processing. Then classify each of the defects using a Densely Connected Convolutional Network (DenseNets) into the following categories, 

1) Missing Hole
2) Mouse Bite
3) Open Circuit
4) Short
5) Spur
6) Spurious Copper

![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/Defects.png?raw=true)


## Classifier Model: 

The DenseNet has a very popular structure with local interconnections as shown below

![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/Dnet.png?raw=true)

In the model proposed, two of these "dense" blocks used are encapsulated between Covolution and Pooling layers as shown below


![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/Dnet2.png?raw=true)

## Results: 

A sample template (left) and defective image (right) are shown below 

![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/TempTest.png?raw=true)

From here, after template matching and some image transformations (detailed in the paper) we localize the defects as shown

![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/PostIP.png?raw=true)

Feeding an ROI drawn around each of these defects to the DenseNet, the final result has the defect labelled along with the confidence

![alt text](https://github.com/MukundSai7907/PCB-Defects-Detection-and-Classification/blob/main/Readme_Images/Result.png?raw=true)

