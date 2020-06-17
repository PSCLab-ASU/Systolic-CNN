# Systolic CNN
This is the source code for the FCCM'20 abstract entitled "Systolic-CNN: An OpenCL-defined Scalable Run-time-flexible FPGA Accelerator Architecture for Accelerating Convolutional Neural Network Inference in Cloud/Edge Computing" (https://ieeexplore.ieee.org/abstract/document/9114649)

## ABSTRACT
This paper presents Systolic-CNN, an OpenCLdefined scalable, run-time-flexible FPGA accelerator architecture, optimized for performing the low-latency, energy-efficient inference of various convolutional neural networks (CNNs) in the context of multi-tenancy cloud/edge computing. Systolic-CNN adopts a highly pipelined and parallelized 1-D systolic array architecture, which efficiently explores both spatial and temporal parallelism for accelerating CNN inference on FPGAs. SystolicCNN is highly scalable and parameterized, which can be easily adapted by users to achieve 100% utilization of the coarsegrained computation resources (i.e., DSP blocks) for a given FPGA. In addition, Systolic-CNN is run-time-flexible, which can be time-shared, in the context of multi-tenancy cloud or edge computing, to accelerate a variety of CNN models at run time without the need of recompiling the FPGA kernel hardware nor reprogramming the FPGA. The experiment results based on an Intel Arria 10 GX FPGA Development board show that Systolic-CNN, when mapped with a single-precision data format, can achieve 100% utilization of the DSP block resource and an average inference latency of 10ms, 84ms, 1615ms, and 990ms per image for accelerating AlexNet, ResNet-50, RetinaNet, and Light-weight RetinaNet, respectively. The peak computational throughput is measured at 80–170 GFLOPS/s across the acceleration of different CNN models.

## How to use
Following steps is recommended to be followed in the given order to run the Systolic CNN on the available FPGA hardware
1. To generate the FPGA hardware, [device codes](conv/conv/conv/device/) need to be compiled using OpenCL SDK. Since we have used [Intel FPGA SDK for OpenCL]18.0 version, [Command](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md#command-to-compile-the-opencl-code)is used to generate the hardware. This step is the most time consuming step and might take 5-6 hrs of the hardware generated based on the Systolic CNN architectural parameters defination. This step generates the generalized CNN hardware which can be used for multiple CNN model such as Alexnet and Resnet-50.
3. After hardware generation, [host file](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/host/src) need to be compiled to generate the executable file inside [bin] folder(https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/bin). This section explains [how to compile the host file](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md#command-to-compile-the-host-code)
4. [Parameter Readme](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/Parameter_Readme.md) can be used to understand the parameter defination used in both device and host code. 
To implement the Systolic-CNN and run Alexnet and Resnet-50, first pre-trained weights and the parameters values for the Alexnet and Resnet-50 CNN model need to be downloaded from [here](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md)

## Tested Boards
All the experiments are conducted based on an Intel Arria 10 GX FPGA Development board that is equipped with an Intel 10AX115S2F45I1SG FPGA and 2GB DDR4 SDRAM with a maximum memory bandwidth of 19.2 GB/s. We have used Intel FPGA SDK for OpenCL version Pro 18.0 for device code compilation and deployment.

## Contributors
- Akshay Dua (@AkshayDua)




