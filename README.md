# Systolic Array CNN

Kernel Design parameters 
1. Device code can be found in /conv/conv/device/folder
2. Two parameters that can be used to scale the design are : pe_num and ll_reuse.
3. Pe_num paramters increases the size of weight buffer.
4. ll_reuse increases the size of input shift register.
5. Convolution is divided into three kernels namely memrd, conv and memwrite
6. Other layers added to the design is Local response normalization, pooling-average and max, eltwise and relu.
7. Kernel parameters does not change the size of other layers. 

Host Design paramters
1. From Host side various parameters are send to different kernel for given operations
2. Parameters send to memrd kernel are 

