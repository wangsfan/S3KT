# S3KT
## Introduction
This is our implementation of our paper *Unsupervised Domain Adaptive Monocular Event Depth Estimation with Style-Structure-Silhouette Knowledge Transfer*. Authors: Jianye Yang, Shaofan Wang, Jiangna Xie, Yanfeng Sun, Baocai Yin, Submitted to IEEE Transactions on Multimedia


**Abstract**:
Event cameras capture asynchronous pixel-level in tensity changes, leading to wide applications of monocular depth estimation under high-speed and low-light environments. Due to the cost of sensor and complex of annotating, the lack of large-scale datasets with depth labels impedes the practical applications of event cameras. While there are few research
 works on the unsupervised domain adaptive monocular event
 depth estimation (UDAMED), the large gap between the intensity
 image domain and the event domain is difficult to fill, since
 monocular intensity images provide unconfident clues of target
 scenarios. Inspired from the idea of style transfer and asymptotic
 knowledge transfer, we we propose the Style-Structure-Silhouette
 Knowledge Transfer framework (dubbed S3KT) for UDAMED.
 The key observation is twofold. For one thing, style transfer can
 generate synthetic intensity images endowed with event-stylized
 augmented knowledge; for another, spatiotemporal dynamic
 contexts of events can be smoothly modeled along different
 representation forms of intensity images and events. Typically,
 S3K consists of an Image Branch, a reconstruction sub-branch
 and a depth sub-branch, which transfer three types of knowledge:
 style, structure and silhouette knowledge asymptotically, and
 align the intensity image domain with the event domain promis
ingly. Extensive experiments on MVSEC and DENSE demonstrate
 that, S3KT achieves satisfactory depth estimation performance
 compared with several unsupervised domain adaptive as well
 as some supervised methods. 
 
 ## Dependencies
- python==3.9
- torch==2.4.1
