# Feature-based-Approaches-to-Multi-View-Image-Stitching
## Samuel Venezia, Sonya Coleman, Dermot Kerr, and John Fegan
This repo contrains the code and data related to the conference paper, Feature-based Approaches to Multi-View Image Stitching.

| Stages        | Algorithm     | Paramters     |
| ------------- | ------------- | ------------- | 
| 1 & 2          | SIFT  | Number of Octave Layers=3, <br>Contrast Threshold=0.04,<br> Edge Threshold=10, <br>Sigma=1.6 |
| 1 & 2           | ORB  | Scale Factor=1.2,<br> Number of levels=8, <br>Edge Threshold=31,<br> First Level=0, <br>WTAK=2,<br> Patch Size=31, <br>Fast Threshold=20 |
| 1 & 2 | BRISK | Detection Threshold=30, <br>Number of Octaves=3,<br> Pattern Scale=1.0 |
| 1 & 2 | AKAZE | Number of Descriptor Channels = 3, <br>Detection Threshold = 0.001,<br> Number of Octaves=4, <br>Number of Octave Layers=4|
| 2 | FREAK | Pattern Scale=22.0, <br>Number of Octaves=4 |
| 1 & 2 | SuperPoint | Non Maximum Suppression radius=4, <br>Keypoint Threshold=0.005, <br>Maximum Keypoints=1024 |
| 2 & 3 | SuperGlue | Sinkhorn Iterations = 20 <br> Match Threshold=0.2 |
| 3 | BF | Distance = Euclidean |
| 3 | BF KNN | Distance = Euclidean, <br>K=2, <br>Matching Distance=0.75 |
| 4 | RANSAC & USAC | Threshold=5.0 <br> Confidence=0.995 |
