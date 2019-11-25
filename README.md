# Systolic Array CNN

# ABSTRACT
Our work presents a generic OpenCL-defined CNN accelerator architecture optimized for FPGA-based real-tim analysis of images on edge. The proposed CNN OpenCL kernel adopts a highly pipelined and parallelized 1-D systolic array architecture, which explores both spatial and temporal parallelism for energy efficiency CNN acceleration on FPGAs. The proposed CNN kernel is highly scalable and parameterized by architecture parameters, namely pe_num and reuse_fac, which can be adapted to achieve 100% utilization of the coarse-grained computation resources (DSP blocks) for a given FPGA. This also makes are design more scalable and can be deployed to multiple FPGA platforms. 
The performance of Alexnet, Resnet-50 has been measured by the proposed CNN kernel on Intel Arria 10 GX1150 FPGA. 

# How to use
Pre-trained weight used for alexnet_weights can be downloaded from https://www.dropbox.com/s/usi9bhlvb9cqt9n/alexnet_weights.tar?dl=0

Kernel Design parameters 
1. Device code can be found in /conv/conv/device/folder
2. Two parameters that can be used to scale the design are : pe_num and ll_reuse.
3. Pe_num paramters increases the size of weight buffer.
4. ll_reuse increases the size of input shift register.
5. Convolution is divided into three kernels namely memrd, mask_read, conv(autorun kernel) and memwrite
6. Other layers added to the design is Local response normalization, pooling-average and max, eltwise and relu.
7. Kernel parameters does not change the size of other layers. 

Host Design paramters
1. From Host side various parameters are send to different kernel for given operations
2. Parameters send to memrd kernel are input feature map, input feature map dimensions( all three dimensions), window paramters window and window2 along x, y axis respectively, pool parameters to determine if layer is pooling and which pooling -average or max and stride parameters stride_conv, stride_conv1 along x, y axis respectively.
3. 

