# Systolic CNN

# ABSTRACT
Our work presents a generic OpenCL-defined CNN accelerator architecture optimized for FPGA-based real-time analysis of images on edge. The proposed CNN OpenCL kernel adopts a highly pipelined and parallelized 1-D systolic array architecture, which explores both spatial and temporal parallelism for energy efficiency CNN acceleration on FPGAs. The proposed CNN kernel is highly scalable and parameterized by architecture parameters, namely pe_num and reuse_fac, which can be adapted to achieve 100% utilization of the coarse-grained computation resources (DSP blocks) for a given FPGA. This also makes are design more scalable and can be deployed to multiple FPGA platforms. 
The performance of Alexnet, Resnet-50 has been measured by the proposed CNN kernel on Intel Arria 10 GX1150 FPGA. 

# How to use
To use the data first pre-trained weights are need to be downloaded from the 
Pre-trained weight used for alexnet_weights can be downloaded from https://www.dropbox.com/s/usi9bhlvb9cqt9n/alexnet_weights.tar?dl=0

Kernel Design parameters 
1. Device code can be found here at the [device folder](Systolic-CNN/conv/conv/conv/device/)


Host Design paramters


