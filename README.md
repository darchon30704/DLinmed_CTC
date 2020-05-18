# DLinmed_3D_DICOM_CT

## Predicting the probability of colorectal malignancy from CT Colonography

A computerized tomography (CT) scan is one of the most widespread imaging examination worldwide for diagnosis and screening many disorders, including adenomas, neoplasma and hemorrhages.
The size of the polyp is used to determine whether immediate polypectomy or polyp surveillance is prefered. Polyp sizes of 6 and 10 mm are used as critical thresholds for clinical decision making.

## CT Colonography Dataset
3,451 series and 941,771 images
69 cases with 6 to 9 mm polyps
35 cases which have at least one > 10 mm polyp and their histological type (836 studies). 

## 3D ResNets
A basic ResNets block consists of two convolutional layers

Each convolutional layer is followed by batch normalization and a ReLU

A shortcut pass connects the top of the block to the layer just before the last ReLU in the block

## Implementation
loss function: Cross Entropy Loss

Optimization: Stochastic Gradient Descent (SGD) with momentum to helps accelerate gradients vectors in the right directions, thus leading to faster converging

The training parameters include a weight decay of 0.0001 and 0.9 for momentum


