# Systolic CNN

# ABSTRACT
This work presents Systolic-CNN, an OpenCLdefined scalable, run-time-flexible FPGA accelerator architecture,optimized for accelerating the inference of various convolutional neural networks (CNNs) in multi-tenancy cloud/edge computing. The existing OpenCL-defined FPGA accelerators for CNN inference are insufficient due to limited flexibility for supporting multiple CNN models at run time and poor scalability resulting in underutilized FPGA resources and limited computational parallelism. Systolic-CNN adopts a highly pipelined and paralleled 1-D systolic array architecture, which efficiently explores both spatial and temporal parallelism for accelerating CNN inference on FPGAs. Systolic-CNN is highly scalable and parameterized, which can be easily adapted by users to achieve 100% utilization of the coarse-grained computation resources (i.e., DSP blocks) for a given FPGA. Systolic-CNN is also run-time-flexible in the context of multi-tenancy cloud/edge computing, which can be time-shared to accelerate a variety of CNN models at run time without the need of recompiling the FPGA kernel hardware nor reprogramming the FPGA. 
# How to use
Following steps is recommended to be followed in the given order to run the Systolic CNN on the available FPGA hardware
1. To run the Systolic-CNN, first pre-trained weights and the parameters values for the Alexnet and Resnet-50 CNN model need to be downloaded from [here](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md)
2. To generate the FPGA hardware, [device code](conv/conv/conv/device/) need to be compiled using OpenCL SDK. Since we have used [Intel FPGA SDK for OpenCL]18.0 version, [Command](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md#command-to-compile-the-opencl-code)is used to generate the hardware. This step is the most time consuming step and might take 5-6 hrs of the hardware generated based on the Systolic CNN architectural parameters defination. This step generates the generalized CNN hardware which can be used for multiple CNN model such as Alexnet and Resnet-50.
3. After hardware generation, [host file](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/host/src) need to be compiled to generate the executable file inside [bin] folder(https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/bin). This section explains [how to compile the host file](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/CNN_models.md#command-to-compile-the-host-code)
4. [Parameter Readme](https://github.com/PSCLab-ASU/Systolic-CNN/blob/master/Readme/Parameter_Readme.md) can be used to understand the parameter defination used in both device and host code. 

# Tested Boards
All the experiments are conducted based on an Intel Arria 10 GX FPGA Development board that is equipped with an Intel 10AX115S2F45I1SG FPGA and 2GB DDR4 SDRAM with a maximum memory bandwidth of 19.2 GB/s. We have used Intel FPGA SDK for OpenCL version Pro 18.0 for device code compilation and deployment.

# Contributors
This work is part of research froup [PSC-lab] (https://ren-fengbo.lab.asu.edu) at Arizona State University.
Main contributors for this project are :
Akshay Dua and Dr. Fengbo Ren

# References
1. D. Wang, K. Xu, and D. Jiang, “Pipecnn: An opencl-based open-source fpga accelerator for convolution neural networks,” in 2017 International Conference on Field Programmable Technology (ICFPT). IEEE, 2017,pp. 279–282
2. X. Wei, C. H. Yu, P. Zhang, Y. Chen, Y. Wang, H. Hu, Y. Liang,
and J. Cong, “Automated systolic array architecture synthesis for high throughput cnn inference on fpgas,” in Proceedings of the 54th Annual Design Automation Conference 2017. ACM, 2017, p. 29.
3. N. Suda, V. Chandra, G. Dasika, A. Mohanty, Y. Ma, S. Vrudhula, J.-s.Seo, and Y. Cao, “Throughput-optimized opencl-based fpga accelerator for large-scale convolutional neural networks,” in Proceedings of the 2016 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays. ACM, 2016, pp. 16–25.






