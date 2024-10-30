# AutoOrgan

In this model, a total of 105 human body CT images were used for bone segmentation training, and the preliminary segmentation results were manually repaired, and the repaired results were re-invested in the model for training to obtain the final model, thus ensuring the accuracy of the segmentation results to the greatest extent. 

Bone segment for CT: 
![Alt text](resources/labels.png)

Please also cite [nnUNet](https://github.com/MIC-DKFZ/nnUNet) since TotalSegmentator is heavily based on it.

### Installation

AutoOrgan works on Centos,and only support GPU.

Install dependencies:

* Python >= 3.9
* [Pytorch](http://pytorch.org/) >= 2.0.0
* nnunetv2 == 2.5.1
  
### Usage

step1: open config.py to set your own input and output path.  

step2: download model file,and set model_path is like '/home/xxx/Dataset105_TotalBone/nnUNetTrainer__nnUNetPlans__3d_fullres'  

step3: to run main.py, wait a while, you'll find result in output folder.  

You can download model file in this link:https://drive.google.com/file/d/1ys89h0it4Lo6e5Y8fyU7SmRf9O41y3pP/view?usp=sharing 
### Class details

|Index|AutoOrgan name|
|:-----|:-----|
1 | left_kneecap |
2 | right_kneecap |
3 | left_tibia |
4 | left_fibula |
5 | left_heel_root |
6 | left_heel_mid |
7 | left_toe_bone |
8 | sternum |
9 | skull |
10 | right_tibia |
11 | right_fibula |
12 | vertebrae_C1 |
13 | vertebrae_C2 |
14 | vertebrae_C3 |
15 | vertebrae_C4 |
16 | vertebrae_C5 |
17 | vertebrae_C6 |
18 | vertebrae_C7 |
19 | vertebrae_L1 |
20 | vertebrae_L2 |
21 | vertebrae_L3 |
22 | vertebrae_L4 |
23 | vertebrae_L5 |
24 | vertebrae_T1 |
25 | vertebrae_T10 |
26 | vertebrae_T11 |
27 | vertebrae_T12 |
28 | vertebrae_T2 |
29 | vertebrae_T3 |
30 | vertebrae_T4 |
31 | vertebrae_T5 |
32 | vertebrae_T6 |
33 | vertebrae_T7 |
34 | vertebrae_T8 |
35 | vertebrae_T9 |
36 | right_heel_root |
37 | right_heel_mid |
38 | right_toe_bone |
39 | rib_left_1 |
40 | rib_left_2 |
41 | rib_left_3 |
42 | rib_left_4 |
43 | rib_left_5 |
44 | rib_left_6 |
45 | rib_left_7 |
46 | rib_left_8 |
47 | rib_left_9 |
48 | rib_left_10 |
49 | rib_left_11 |
50 | rib_left_12 |
51 | rib_right_1 |
52 | rib_right_2 |
53 | rib_right_3 |
54 | rib_right_4 |
55 | rib_right_5 |
56 | rib_right_6 |
57 | rib_right_7 |
58 | rib_right_8 |
59 | rib_right_9 |
60 | rib_right_10 |
61 | rib_right_11 |
62 | rib_right_12 |
63 | left_scapula |
64 | right_scapula |
65 | left_clavicle |
66 | right_clavicle |
67 | left_thighbone |
68 | right_thighbone |
69 | left_hip |
70 | right_hip |
71 | sacrum |
72 | upper limb |
