# Feature-based-Approaches-to-Multi-View-Image-Stitching
## Samuel Venezia, Sonya Coleman, Dermot Kerr, and John Fegan
This repo contrains the code and data related to the conference paper, Feature-based Approaches to Multi-View Image Stitching.

\begin{tabular}{|l|c|c|} \hline 
          Stages&Algorithms& Parameters\\ \hline 
          1 \& 2&SIFT& \makecell{Number of Octave Layers=3, \\Contrast Threshold=0.04,\\ Edge Threshold=10, \\Sigma=1.6}\\ \hline 
          1 \& 2&ORB & \makecell{Scale Factor=1.2,\\ Number of levels=8, \\Edge Threshold=31,\\ First Level=0, \\WTAK=2,\\ Patch Size=31, \\Fast Threshold=20} \\ \hline 
          1 \& 2&BRISK& \makecell{Detection Threshold=30, \\Number of Octaves=3,\\ Pattern Scale=1.0} \\ \hline 
          1 \& 2&AKAZE& \makecell{Number of Descriptor Channels = 3, \\Detection Threshold = 0.001,\\ Number of Octaves=4, \\Number of Octave Layers=4}\\ \hline 
          2&FREAK& \makecell{Pattern Scale=22.0, \\Number of Octaves=4} \\ \hline 
          1 \& 2&SuperPoint& \makecell{Non Maximum Suppression radius=4, \\Keypoint Threshold=0.005, \\Maximum Keypoints=1024 }\\ \hline 
          2 \& 3&SuperGlue& \makecell{Sinkhorn Iterations=20, \\Match Threshold=0.2} \\ \hline 
          3&\makecell{BF}& \makecell{Distance = Euclidean}\\ \hline 
 3& BF KNN&\makecell{Distance = Euclidean, \\K=2, \\Matching Distance=0.75}\\ \hline 
 4& \makecell{RANSAC \& \\USAC}&\makecell{Threshold=5.0,  \\Confidence=0.995} \\\hline
