# CNN models
The pre-trained Alexnet and Resnet-50 models used in our work are available at the [caffe2 GitHub repostory](https://github.com/facebookarchive/models).

# Layer-wise parameter definition 
To run the Alexnet and Resnet-50 using Systolic-CNN on a given FPGA, layer-wise parameter definition of the respective CNN model is required. This can be downloaded from the following [Dropbox Folder](https://www.dropbox.com/sh/lt3ytk6zzq5qxsr/AADzwfkLFSJrd7Ld3LRQmn-ya?dl=0).


# Library needed to run Systolic-CNN models
1. To generate the FPGA kernel hardware for Systolic-CNN, an OpenCL SDK is needed. We used [Intel FPGA SDK for OpenCL](https://www.intel.com/content/www/us/en/software/programmable/sdk-for-opencl/overview.html) 18.0 pro version. 
2. To run the demo, OpenCV is needed for reading the jpg image, pre-processing, post-processing and displaying the image and other results via a GUI. 

# Command to compile the device kernel
1. The command for compiling the [device kernel](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/device) in Intel FPGA SDK for OpenCL is "aoc -v -g -I $INTELFPGAOCLSDKROOT/include/kernel_headers/ gen_conv.cl".
3. This should generate a binary file in .aocx, .aoco, or .aocr extension. All these files need to be copied to the bin folder to run the Systolic-CNN for different CNN models

# Command to compile the host kernel
[Host code](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/host/src) is specific to the given CNN model and is written in C++. We have provided support for the Alexnet and Resnet-50 model with conv_alexnet.cpp and conv_resnet.cpp file inside the host folder. To avoid the conflict of generating multiple or wrong executable files, it is recommended to have only one C++ file inside the folder and to either remove the other C++ filer or copy to a backup folder. Following steps are required to generate the executable file
1. [Makefile](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv) is used to generate the executable file at the bin location.
2. To generate the executable file, use " make clean; make". This will generate the executable file specific to the target CNN model.
