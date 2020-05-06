# Convolution models locations
We have used pre-trained CNN models, which are trained using Caffe2 deep learning framework. Alexnet and Resnet-50 CNN model defination are available at the [caffe2 github repostory](https://github.com/facebookarchive/models) 

# Weights location 
To run the Alexnet and Resnet-50 CNN model using Systolic-CNN on given FPGA, along with pre-trained model, parameter defination per layer of the respective CNN model is required. This can downloaded from our own [ModelZoo](https://www.dropbox.com/home/SystolicArrayCNN)


# Library needed to run Systolic-CNN models
1. Hardware for the Systolic-CNN model is generated using the OpenCL based SDK. We have used [Intel FPGA SDK for OpenCL](https://www.intel.com/content/www/us/en/software/programmable/sdk-for-opencl/overview.html) 18.0 version to generate the FPGA hardware. 
2. To run the demo, which requires reading the jpg image, pre-processing, post processing and displaying the image and other results on the gui, requires the support of OpenCV

# Command to compile the OpenCL code
1. [Device code](https://github.com/PSCLab-ASU/Systolic-CNN/tree/master/conv/conv/conv/device) is used to generate the FPGA hardware
2. Command for Intel FPGA SDK for OpenCL is  " aoc -v -g -I $INTELFPGAOCLSDKROOT/include/kernel_headers/ gen_conv.cl
3. This should generate binary file of .aocx, .aoco, .aocr extension. All these files need to be copied to bin folder to run the Systolic-CNN for the CNN models
