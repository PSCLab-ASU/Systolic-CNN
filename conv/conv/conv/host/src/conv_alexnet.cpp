// faster_rcnn.cpp
//

#define AOCL_ALIGNMENT 64
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<cstddef>
#include <time.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <stdlib.h>
//#include "data.h"
//#include "data_res2_0_branch.h"
//#include "data_res2_0_branch2a.h"
//#include "data_res2_1_branch.h"
//#include "data_res2_2_branch.h"
//#include "data_res3_0_branch.h"
//#include "data_res3_1_branch.h"
//#include "data_res3_2_branch.h"
//#include "data_res3_3_branch.h"
//#include "data_res4_0_branch.h"
//#include "data_res4_1_branch.h"
//#include "data_res4_0_1branch.h"
//#include "data_res4_2_branch.h"
//#include "data_res4_3_branch.h"
//#include "data_res4_4_branch.h"
//#include "data_res4_5_branch.h"
//#include "data_convrpn_branch.h"
//#include "data_rpn_cls.h"
//#include "data_generate_branch.h"
//#include "data_res5_0_branch.h"
//#include "data_res5_1_branch.h"
//#include "data_res5_2_branch.h"
//#include "data_cls_score.h"
//#include "data_bbox_pred.h"
//#include "data1.h"
//#include "data_new.h" 
//#include "data_new_stride.h"
//#include "data_new1.h"          /// done for new type of convlution ///
//#include "data_new_stride11.h"
//#include "data_shift5.h"   // for without stride convolution //
//#include "data1_shift5.h" // for stride convolution 
//#include "data2_shift5.h"
//#include "data3_shift5.h"      /// 5 * 5 * 96///
//#include "data4_shift5.h"       ///// 11 * 11*3 ///
//#include "data5_shift5.h"       ///// 3 * 3* 256 ///
//#include "data6_shift5.h"       ///// 3 * 3 * 192 ///
//#include "data7_shift5.h"       ///// 3 * 3 * 192 ///
//#include "data8_shift5.h"         ///// fc1 ///
#include "img9.h"   // park_bench.jpg
#include "img6.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cvwimage.h"
#include "opencv/cxcore.h"
#include "opencv/cvconfig.h"
#include "opencv/cvaux.h"
#include "opencv/cvcompat.h"
#include "opencv/cvtypes.h"
#include "opencv/cxoperations.hpp"
#include "opencv/cvver.h"
#include "opencv/cvvidsurv.hpp"
#include "opencv/cxerror.h"
#include "opencv/cxflann.h"
#include "opencv/cxmisc.h"
#include "opencv/ml.h"
#include "opencv/cxmat.hpp"
#include "opencv/cxoperations.hpp"
#include "img8.h"
#include "conv1_w.h"
#include "pool_params.h"
#include "conv2_w.h"
#include "lrn_params.h"
#include "conv3_w.h"
#include "conv4_w.h"
#include "conv5_w.h"
#include "fc6.h"
#include "fc7.h"
#include "fc8.h"
#include <python2.6/Python.h>
#include "label.h"
#include "label_check.h"
using namespace aocl_utils;
using namespace cv;
using namespace std;
#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif
////inputs //////////////
// Constants
//const unsigned int inputSignalWidth  = 8;
//const unsigned int inputSignalHeight = 8;
bool init_opencl();

//cl_uint inputSignal[inputSignalWidth][inputSignalHeight] =
//{
//	{3, 1, 1, 4, 8, 2, 1, 3},
//	{4, 2, 1, 1, 2, 1, 2, 3},
//	{4, 4, 4, 4, 3, 2, 2, 2},
//	{9, 8, 3, 8, 9, 0, 0, 0},
//	{9, 3, 3, 9, 0, 0, 0, 0},
//	{0, 9, 0, 8, 0, 0, 0, 0},
//	{3, 0, 8, 8, 9, 4, 4, 4},
//	{5, 9, 8, 1, 8, 1, 1, 1}
//};

//const unsigned int outputSignalWidth  = 6;
//const unsigned int outputSignalHeight = 6;

//cl_uint outputSignal[outputSignalWidth][outputSignalHeight];

//const unsigned int maskWidth  = 3;
//const unsigned int maskHeight = 3;
#define STRING_BUFFER_LEN 1024

//cl_uint mask[maskWidth][maskHeight] =
//{
//	{1, 1, 1}, {1, 0, 1}, {1, 1, 1},
//};
//added new to find platform
cl_platform_id platform = NULL;
cl_platform_id* platforms;
cl_uint numPlatforms;
static void display_device_info( cl_device_id device );
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
/// inputs ///////////////
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
void cleanup();

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{          
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

///
//	
int main(int argc, char** argv)
{   /// added to read data file 
//extern CL_API_ENTRY cl_int CL_API_CALL;
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue1;
        cl_command_queue queue2;
        cl_command_queue queue3;  
        cl_command_queue queue4;
        cl_command_queue queue5;
        cl_command_queue queue6;  
        cl_command_queue queue7;
        cl_command_queue queue8;   
	cl_program program;
	cl_kernel kernel1;
        cl_kernel kernel2;
        cl_kernel kernel3;  
	cl_mem inputSignalBuffer_r[15];
        cl_mem inputSignalBuffer_g;
        cl_mem inputSignalBuffer_b;
        cl_mem pool1Buffer;
        cl_mem res2_0_branch1_bnBuffer;
	cl_mem maskBuffer_r;
        cl_mem maskBuffer_g;
        cl_mem maskBuffer_b;
        cl_mem biasBuffer;
        cl_mem res2_0_branch1_w_Buffer;///names tells about the variable name
        cl_mem res2_0_branch1_bn_b_Buffer;
        cl_mem res2_0_branch2a_w_Buffer;
        cl_mem res2_0_branch2a_bn_b_Buffer;
        cl_mem res2_0_branch2b_bn_b_Buffer;
        cl_mem res2_0_branch2b_w_Buffer;
        cl_mem res2_0_branch2b_Buffer;
        cl_mem res2_0_branch2a_Buffer;
        cl_mem res2_0_branch2c_bn_Buffer;
        cl_mem res2_0_branch2c_bn_b_Buffer;
        cl_mem res2_0_branch2c_w_Buffer;
        cl_mem res2_1_branch2a_w_Buffer;
        cl_mem res2_1_branch2a_Buffer;
        cl_mem res2_1_branch2a_bn_b_Buffer;
        cl_mem res2_1_branch2b_Buffer;
        cl_mem res2_1_branch2b_bn_b_Buffer;
        cl_mem res2_1_branch2b_w_Buffer;
        cl_mem res2_1_branch2c_bn_b_Buffer;
        cl_mem res2_1_branch2c_w_Buffer;
        cl_mem res2_1_branch2c_bn_Buffer;
        cl_mem res2_2_branch2a_w_Buffer;
        cl_mem res2_2_branch2a_bn_b_Buffer;
        cl_mem res2_2_branch2a_Buffer;
        cl_mem res2_2_branch2b_Buffer;
        cl_mem res2_2_branch2b_bn_b_Buffer;
        cl_mem res2_2_branch2b_w_Buffer;
        cl_mem res2_2_branch2c_bn_b_Buffer;
        cl_mem res2_2_branch2c_w_Buffer;
        cl_mem res2_2_sum_Buffer; /// no need of res2_2_branch2c_bn here
        cl_mem res3_0_branch2a_w_Buffer;
        cl_mem res3_0_branch2a_bn_b_Buffer;
        cl_mem res3_0_branch1_w_Buffer;
        cl_mem res3_0_branch1_bn_b_Buffer;
        cl_mem res3_0_branch2a_Buffer;
        cl_mem res3_0_branch1_bn_Buffer;
        cl_mem res3_0_branch2b_bn_b_Buffer;
        cl_mem res3_0_branch2b_w_Buffer;
        cl_mem res3_0_branch2c_bn_b_Buffer;
        cl_mem res3_0_branch2c_w_Buffer;
        cl_mem res3_0_branch2c_bn_Buffer;
        cl_mem res3_0_branch2b_Buffer;
        cl_mem res3_1_branch2a_w_Buffer;
        cl_mem res3_1_branch2a_bn_b_Buffer;
        cl_mem res3_1_branch2a_Buffer;
        cl_mem res3_1_branch2b_w_Buffer;
        cl_mem res3_1_branch2b_Buffer;
        cl_mem res3_1_branch2b_bn_b_Buffer;
        cl_mem res3_1_branch2c_bn_b_Buffer;
        cl_mem res3_1_branch2c_w_Buffer;
        cl_mem res3_1_branch2c_bn_Buffer;
        cl_mem res3_2_branch2a_w_Buffer;
        cl_mem res3_2_branch2a_bn_b_Buffer;
        cl_mem res3_2_branch2a_Buffer;
        cl_mem res3_2_branch2b_bn_b_Buffer;
        cl_mem res3_2_branch2b_w_Buffer;
        cl_mem res3_2_branch2b_Buffer;
        cl_mem res3_2_branch2c_bn_b_Buffer;
        cl_mem res3_2_branch2c_w_Buffer;
        cl_mem res3_2_branch2c_bn_Buffer;
        cl_mem res3_3_branch2a_w_Buffer;
        cl_mem res3_3_branch2a_bn_b_Buffer;
        cl_mem res3_3_branch2a_Buffer;
        cl_mem res3_3_branch2b_bn_b_Buffer;
        cl_mem res3_3_branch2b_w_Buffer;
        cl_mem res3_3_branch2b_Buffer;
        cl_mem res3_3_branch2c_w_Buffer;
        cl_mem res3_3_branch2c_bn_b_Buffer;
        cl_mem res3_3_sum_Buffer; //// no need of res3_3_branch2c_bn here//
        cl_mem res3_4_branch2a_Buffer;
        cl_mem res4_0_branch2a_w_Buffer;
        cl_mem res4_0_branch2a_bn_b_Buffer;
        cl_mem res4_0_branch2a_Buffer;
        cl_mem res4_0_branch1_w_Buffer;
        cl_mem res4_0_branch2b_bn_b_Buffer;
        cl_mem res4_0_branch2b_Buffer;
        cl_mem res4_0_branch1_bn_b_Buffer;
        cl_mem res4_0_branch2b_w_Buffer;
        cl_mem res4_0_branch2c_bn_b_Buffer;
        cl_mem res4_0_branch2c_w_Buffer;
        cl_mem res4_0_branch2c_bn_Buffer;
        cl_mem res4_1_branch2a_w_Buffer;
        cl_mem res4_1_branch2a_Buffer;
        cl_mem res4_1_branch2a_bn_b_Buffer;
        cl_mem res4_1_branch2b_bn_b_Buffer;
        cl_mem res4_1_branch2b_w_Buffer;
        cl_mem res4_1_branch2b_Buffer;
        cl_mem res4_1_branch2c_bn_b_Buffer;
        cl_mem res4_1_branch2c_w_Buffer;
        cl_mem res4_1_branch2c_bn_Buffer;
        cl_mem res4_2_branch2a_w_Buffer;
        cl_mem res4_2_branch2a_Buffer;
        cl_mem res4_2_branch2a_bn_b_Buffer;
        cl_mem res4_2_branch2b_w_Buffer;
        cl_mem res4_2_branch2b_Buffer;
        cl_mem res4_2_branch2b_bn_b_Buffer;
        cl_mem res4_2_branch2c_bn_b_Buffer;
        cl_mem res4_2_branch2c_w_Buffer;
        cl_mem res4_2_branch2c_bn_Buffer;
        cl_mem res4_3_branch2a_w_Buffer;
        cl_mem res4_3_branch2a_Buffer;
        cl_mem res4_3_branch2a_bn_b_Buffer;
        cl_mem res4_3_branch2b_w_Buffer;
        cl_mem res4_3_branch2b_Buffer;
        cl_mem res4_3_branch2b_bn_b_Buffer;
        cl_mem res4_3_branch2c_bn_b_Buffer;
        cl_mem res4_3_branch2c_w_Buffer;
        cl_mem res4_3_branch2c_bn_Buffer;
        cl_mem res4_4_branch2a_w_Buffer;
        cl_mem res4_4_branch2a_bn_b_Buffer;
        cl_mem res4_4_branch2a_Buffer;
        cl_mem res4_4_branch2b_bn_b_Buffer;
        cl_mem res4_4_branch2b_w_Buffer;
        cl_mem res4_4_branch2c_bn_b_Buffer;
        cl_mem res4_4_branch2b_Buffer;
        cl_mem res4_4_branch2c_bn_Buffer;
        cl_mem res4_4_branch2c_w_Buffer;
        cl_mem res4_5_branch2a_w_Buffer;
        cl_mem res4_5_branch2a_Buffer;   
        cl_mem res4_5_branch2a_bn_b_Buffer;
        cl_mem res4_5_branch2b_w_Buffer;
        cl_mem res4_5_branch2b_bn_b_Buffer;
        cl_mem res4_5_branch2b_Buffer;
        cl_mem res4_5_branch2c_bn_Buffer;
        cl_mem res4_5_branch2c_w_Buffer;
        cl_mem res4_5_branch2c_bn_b_Buffer;
        cl_mem conv_rpn_w_Buffer;
        cl_mem conv_rpn_b_Buffer;
        cl_mem conv_rpn_Buffer;
        cl_mem rpn_cls_logits_b_Buffer;
        cl_mem rpn_cls_logits_w_Buffer;
        cl_mem rpn_cls_probs_Buffer;
        cl_mem rpn_bbox_pred_w_Buffer;
        cl_mem rpn_bbox_pred_b_Buffer;
        cl_mem rpn_bbox_pred_Buffer;
        cl_mem anchor_Buffer;  /// after data preparation from all_anchor, all_anchor is send 
        cl_mem im_info_Buffer;
        cl_mem rpn_rois_Buffer;
        cl_mem pool5_Buffer;
        cl_mem res5_0_branch2a_w_Buffer;
        cl_mem res5_0_branch2a_bn_b_Buffer;
        cl_mem res5_0_branch2a_Buffer;
        cl_mem res5_0_branch2b_w_Buffer;
        cl_mem res5_0_branch2b_bn_b_Buffer;
        cl_mem res5_0_branch2b_Buffer;
        cl_mem res5_0_branch2c_bn_Buffer;
        cl_mem res5_0_branch2c_bn_b_Buffer;
        cl_mem res5_0_branch2c_w_Buffer;
        cl_mem res5_0_branch1_w_Buffer;
        cl_mem res5_0_branch1_bn_b_Buffer;
        cl_mem res5_0_branch1_bn_Buffer;
        cl_mem res5_1_branch2a_w_Buffer;
        cl_mem res5_1_branch2a_bn_b_Buffer;
        cl_mem res5_1_branch2a_Buffer;
        cl_mem res5_1_branch2b_w_Buffer;
        cl_mem res5_1_branch2b_bn_b_Buffer;
        cl_mem res5_1_branch2b_Buffer;
        cl_mem res5_1_branch2c_w_Buffer;
        cl_mem res5_1_branch2c_bn_b_Buffer;
        cl_mem res5_1_branch2c_bn_Buffer;
        cl_mem res5_2_branch2a_Buffer;
        cl_mem res5_2_branch2a_w_Buffer;
        cl_mem res5_2_branch2a_bn_b_Buffer;
        cl_mem res5_2_branch2b_Buffer;
        cl_mem res5_2_branch2b_w_Buffer;
        cl_mem res5_2_branch2b_bn_b_Buffer;
        cl_mem res5_2_branch2c_w_Buffer;
        cl_mem res5_2_branch2c_bn_b_Buffer;
        cl_mem res5_2_branch2c_bn_Buffer;
        cl_mem res5_pool_Buffer;
        cl_mem cls_score_w_Buffer;
        cl_mem cls_score_b_Buffer;
        cl_mem cls_score_Buffer;
        cl_mem bbox_pred_w_Buffer;
        cl_mem bbox_pred_b_Buffer;
        cl_mem bbox_pred_Buffer;
        cl_mem pred_bbox_Buffer;
        cl_mem cls_prob_Buffer;
        cl_mem output_Buffer;
        cl_mem bias_Buffer;
        cl_mem output_Buffer1;
        cl_mem inputSignalBuffer_r1;
        cl_mem maskBuffer_r1;
        cl_mem outputBuffer_pool_1;
        cl_mem outputBuffer_lrn_1;
        cl_mem outputBuffer_conv2;
        cl_mem bias_Buffer2; 
        cl_mem maskBuffer_2;
        cl_mem outputBuffer_lrn_2;
        cl_mem outputBuffer_pool_2;
        cl_mem outputBuffer_conv4; 
        cl_mem outputBuffer_conv3_1; 
        cl_mem outputBuffer_conv3;  
        cl_mem maskBuffer_3;
        cl_mem bias_Buffer3; 
        cl_mem outputBuffer_conv5; ////for convolution 4 //// 
        cl_mem maskBuffer_4;
        cl_mem bias_Buffer4; 
        cl_mem outputBuffer_conv6; ////for convolution 5 //// 
        cl_mem maskBuffer_5;
        cl_mem bias_Buffer5; 
        cl_mem outputBuffer_pool_3;
        cl_mem maskBuffer_6; 
        cl_mem bias_Buffer6;
        cl_mem outputBuffer_fc_6;
        cl_mem outputBuffer_fc1;  
        cl_mem maskBuffer_7; 
        cl_mem bias_Buffer7;
        cl_mem outputBuffer_fc2;  
        cl_mem outputBuffer_fc4;  
       
        cl_mem maskBuffer_8; 
        cl_mem bias_Buffer8;
        cl_mem outputBuffer_fc3;  
        cl_mem outputBuffer_fc5;
        cl_mem outputBuffer_conv7;
       //float res5_pool[kernel_number_8_2][pool8_height/7][pool8_width/7];
       //float pred_bbox[pool_height10][pool_width10];
       //float cls_prob[pool_height9][pool_width9];
           //   float res3_4_branch2a[kernel_number_4_1][pool2_height][pool1_width];
//void *inputSignal_r = NULL;
//posix_memalign ( &inputSignal_r, AOCL_ALIGNMENT, input_width * inputSignalWidth * inputSignalWidth);

//for(int w=0;w< height; w++)
{
for(int i=0;i< input_width;i++)
 {
  for (int j=0;j<inputSignalHeight;j++)
  {
 //   for(int r=0;r<inputSignalWidth1;r++)
     {
      // inputSignal_r1[i][j][r]= random()%10;
        }
   }
}
}
//for(int w =0 ;w< input_height; w++)
{
//for(int i=0;i< input_width;i++)
 {
 // for (int j=0;j<maskHeight;j++)
  {
   // for(int r=0;r<maskWidth;r++)  
     {
     //  mask_1[w][i][j][r]= random()%10;
        }
   }
}
}

//for(int w=0;w< height;w++)
{
// for(int i=0;i< input_width;i++)
 {
  //for(int r=0;r<maskHeight;r++)
  {
    //printf("[");
   //for(int j=0;j<maskWidth;j++)
     {
  
  // printf(" %f ,", mask_1[0][i][r][j]);
   }
 //printf(" \n  \n");
 }
// printf("\n, \n");
}
//printf("\n, \n");
}


const int fc_n_1 = 1; ////1 for convolution 4 for fully connected
const int fc_n = 0;   //// 0 for convolution 1 for fully connected //
const int fc_param_mask = 1;//  0 for convolution  and 1 for stride_convolution // or in new case is always 1 let's see // i guess it need to be 24 but let's see
const int fc_param_mask_2 = 1;  // / 4 for fully connected and 1 for convoiton , 2 for stride of 2 convolution // or for new case it needs to be 1
const int fc_param_mask_3 = 1;  /// for new convolutin it need to be 1
const int fc_param_mask_fc = 0;//  0 for convolution  and 1 for stride_convolution // or in new case is always 1 let's see // i guess it need to be 24 but let's see
const int fc_param_mask_2_fc = 0;  // / 4 for fully connected and 1 for convoiton , 2 for stride of 2 convolution // or for new case it needs to be 1
const int fc_param_mask_3_fc = 0; 
const int fc_check = 1; ///  0 for convolution that requires stride but for other it is 1  ///
const int stride_write = 3;   
const int stride_write1 = 1;  // not needed currently ///
const int sum_param = 1 ; //// only used in resnet when it is 1 that no addition, when it is 0 that means addition
const int avg_pool = 0;
  printf("data r recieved is \n");
//for(int w=0;w< height;w++)
{
 //for(int i=0;i< input_width;i++)
 {
 // for(int r=0;r<inputSignalHeight;r++)
  {
    //printf("[");
   //for(int j=0;j<inputSignalWidth;j++)
     {
  
 //  printf(" %f ,", inputSignal_r1[i][r][j]);
   }
 //printf(" \n  \n");
 }
 //printf("\n, \n");
}
//printf("\n, \n");
}
//// reorder input //////////////
const int image_num = 3;
Mat image, image1[image_num+3], image2, image3, image5[image_num];
float inputSignal_r7[3][227][227], inputSignal_r8[227][227][16], inputSignal_r9[3][227][227], inputSignal_r10[227][227][16],inputSignal_r11[227][227][16],inputSignal_r12[227][227][16],inputSignal_r13[227][227][16];
image = imread("/home/user/adua5/alexnet_check/conv_exp/alexnet_final_deal/conv/conv/host/src/mouse.jpg");
image2 = imread("/home/user/adua5/alexnet_check/conv_exp/alexnet_final_deal/conv/conv/host/src/school_bus.jpg");
image3 = imread("/home/user/adua5/alexnet_check/conv_exp/alexnet_final_deal/conv/conv/host/src/eraser.jpg");
image5[0] = imread("/img_location/espresso.jpg");
image5[1] = imread("/img_location/matchstick.jpg");
image5[2] = imread("img_location/park_bench.jpg");
if(image.empty())
{
  printf(" error in loading image ");
}
else{
namedWindow( "Display Image", 227*227);
resize(image,image1[0],Size(227,227),0,0);
resize(image2,image1[1],Size(227,227),0,0);
resize(image3,image1[2],Size(227,227),0,0);
for(int d =0 ;d< image_num;d++)
{
resize(image5[d],image1[d+3],Size(227,227),0,0);
}
//resize(image3,image1[2],Size(227,227),0,0);
//image1.convertTo(image2, CV_32FC3);
 //cout << (float) image1.at<Vec3b>(0,1)[0] << "" "" << endl;
  for(int i = 0; i<227; i++)
    {
      for(int j=0; j<227;j++)
         {
           for(int k =0; k<3;k++)
            {
                 inputSignal_r1[k][i][j] = (float) image1[0].at<Vec3b>(i,j)[k];
                 inputSignal_r3[k][i][j] = (float) image1[1].at<Vec3b>(i,j)[k]; 
                 inputSignal_r5[k][i][j] = (float) image1[2].at<Vec3b>(i,j)[k]; 
              
           }
       }      
     }    

 printf("true \n");

}
if(stride == 1)
{
for(int i =0; i<input_width_1;i++)
 {
   for(int j =0; j<inputSignalHeight_1;j++)
    {
      for(int k =0 ;k< inputSignalWidth_1;k++)
       {
         inputSignal_r[j][k][i] = inputSignal_r1[i][j][k]; 
        }
      }
    }
 }
else
{
 
for(int i =0; i< input_width_1;i++)
 {
   for(int j =0 ; j< inputSignalHeight_1; j++)
    {
     for(int k =0 ; k< inputSignalWidth_1_1; k++)
      {  
         for(int r =0; r<4;r++)
           {
         inputSignal_r[j][k][i + r * input_width_1] = inputSignal_r1[i][j][k *4+ r]; 
         inputSignal_r4[j][k][i + r * input_width_1] = inputSignal_r3[i][j][k *4+ r];   
          inputSignal_r6[j][k][i + r * input_width_1] = inputSignal_r5[i][j][k *4+ r]; 
       
          }
     
      }
     }
  }
 }

for(int i=0;i<inputSignalHeight_1;i++)
 {
  for(int r=0;r<inputSignalWidth_1;r++)
  {
    //printf("[");
   for(int j=0;j< 16;j++)
     {
  
  // printf(" %f ,", inputSignal_r[i][r][j]);
   }
 //printf(" \n  \n");
 }
// printf("\n, \n");
}
//printf("\n, \n");



//printf("data g recieved is \n");
  

       
//for(r=3;r<inputSignalHeight-3;r++)
  //{
    //printf("[");
   //for(j=3;j<inputSignalWidth-3;j++)
     //{
       //printf(" %f ,", inputSignal_g[r][j]);
     //}
      //printf("],\n");
   //} 
printf("data b recieved is \n");
////reodering inoput ///
int flag = 0;
int flag1 =0;
/////reordering of mask ////
//printf(" valueof data is %f \n", mask_1[16][0][0][0]);
if( reorder ==1)
{
int e =0 ;
//if( stride !=4 )
{
printf(" true\n");

printf(" \nreordered mask is \n");
for(int y =0;y<height;y++)  /// used to be height but now it should be height/4
{

   //for(int j =0; j< 1;j++)
    {
      //for(int k =0;k< 1;k++)
       {
       //  for(int w =0 ;w<4;w++)
          {
       // for(int i =0;i<ll* 2;i++)
         { 
        //for(int  r =0;r<input_width1;r++)
         {
      //  printf("  %f  ", mask_r[y][j][k][i][r]); 
       }
    // printf("\n\n ");
       }
    // printf("\n\n");
    }
   //printf("\n\n"); 
 }
 //printf("\n\n\n");
}
}


 }

for(int y =0;y<height_2 * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2;k++)
      {
        for(int r =0;r<maskwidth_2;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
       conv2_w_1[y][z][k][r][i][j] = conv2_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}

for(int y =0;y<height_3 * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_3; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_3;k++)
      {
        for(int r =0;r<maskwidth_3;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
       conv3_w_1[y][z][k][r][i][j] = conv3_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_4 * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_4; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_4;k++)
      {
        for(int r =0;r<maskwidth_4;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
       conv4_w_1[y][z][k][r][i][j] = conv4_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_5 * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_5; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_5;k++)
      {
        for(int r =0;r<maskwidth_5;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
       conv5_w_1[y][z][k][r][i][j] = conv5_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}


//else
{
for(int y =0;y<height* 4;y++)           // used to be height but now it should be height/2 // for n height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
  for(int j =0; j< input_width_1;j++)
   {
    for(int k =0;k< maskheight_1;k++)
      {
        for(int r =0;r<maskwidth_1_1;r++)
         {
           for(int z =0 ; z<4;z++)
             {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
            if( y ==0 && w ==0 && i ==0 && j ==0 && k ==0 && r ==0 && z ==0)
             {
                 printf(" value of e is %d %f", e,  conv1_w[y * ll * 2 + e  +  i][j][k][r* 4 + z] );
              }
            conv1_w_1[y][k][r][i][j + z * input_width_1] = conv1_w[y * ll/4 +  i][j][k][r* 4 + z];
            if( r == maskwidth_1_1 -1 && z ==2)
              {
                z++;  /// need to be done for only 11x11 convolution 
              } 
         }
        }
     }
   }
  }
  }
 }
 }

for(int y =0;y<input_height_8/4;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<4;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< masklength_8/16; z++)
     {
  for(int j =0; j< 16;j++)
   {
    //for(int k =0;k< 16;k++)
      {
        for(int r =0;r<1;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
       fc8_w_1[y*4][z][w][j]  = fc8_w[y* 4 + w][z*16 + j];
         }
     }
   }
  }
  }
 }
}


for(int y =0;y<input_height_7/4;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<4;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< masklength_7/16; z++)
     {
  for(int j =0; j< 16;j++)
   {
    //for(int k =0;k< 16;k++)
      {
        for(int r =0;r<1;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);
              }
           fc7_w_1[y*4][z][w][j]  = fc7_w[y* 4 + w][z*16 + j];
         }
     }
   }
  }
  }
 }
}

for(int y =0;y<height_6_1/4;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<4;w++)
    {
//for(int i=0;i<4;i++)  // replaces instead of ll 
 {
  // for(int z =0; z< masklength_6/16; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< 16;k++)
      {
        for(int r =0;r<36;r++)
         {
            flag = w & 0x01;
            flag1 = (~flag & 0x01);   
            e = 16 * flag + ( w-1) * flag * 2 +  (w) * 2 * flag1;
  //          if( y ==0 && w ==1 && i ==0 && j == 0 && k ==0 && r ==0)
             {
    //            printf(" value is %d %f \n ", e, mask_1[y * ll * 2 + e  +  i][j][k][r]);	
              }
           fc6_w_1[y*4][k][r][w][j]  = fc6_w[y*4 + w][k * 16 * 36 + r  +j * 36];
         }
     }
   }
  }
  }
 }
}

printf(" value of input is %d %d %d  %f %f \n", input_width_1, input_width,  input_width1,  inputSignal_r[0][12][0], inputSignal_r[0][0][216]);
printf(" value of conv is %d, %f %f %f  \n",height/2,  conv1_w_1[0][0][0][4][0],conv1_w[0][0][0][0], conv1_w[0][0][0][1]);
//printf(" value of data is %f\n", mask_r[0][0][0][4][0]);
printf(" \nreordered mask is \n");
//for(int i=0;i<maskHeight;i++)
 {
 // for(r=0;r<maskWidth1;r++)
  {
    //printf("[");
   //for(j=0;j< 16;j++)
     {
  
   //printf(" %f ,", mask_r[0][i][r][0][j]);
   }
// printf(" \n  \n");
 }
// printf("\n, \n");
}
//printf("\n, \n");

}
//for(r=3;r<inputSignalHeight-3;r++)
  //{
    // printf("[");
   //for(j=3;j<inputSignalWidth-3;j++)
     //{
      // printf(" %f ,", inputSignal_b[r][j]);
     //}
     //printf("],\n");
   //} 
//anchor data preparation for  generate proposals algorithm///
//since it is one time thing do not need to be done for multiple times ///
//float shift_x[pool_output5*pool_output5], shift_y[pool_output5*pool_output5],shifts[pool_output5* pool_output5][kernel_number_7_2];
//float all_anchors[pool_output5*pool_output5][kernel_number_7_1][kernel_number_7_2];
//float feat_stride= 16.0; //determined by the weight
//for (int i=0; i< pool_output5;i++)
//{
   //for (int j=0;j<pool_output5;j++)
  //{
    //shift_x[i* pool_output5+ j]= j * feat_stride;
    //shift_y[i* pool_output5+ j]= i * feat_stride;
//}
//}

//for(int j=0;j<pool_output5* pool_output5;j++)
  //  {
    //  shifts[j][0]= shift_x[j];
     // shifts[j][1]= shift_y[j];
      //shifts[j][2]= shift_x[j];
      //shifts[j][3]= shift_y[j];
//}
//for(int i=0;i<pool_output5* pool_output5;i++)
//{
 // for(int j=0;j<kernel_number_7_1;j++)  
   //{ 
     //for(int k=0;k<kernel_number_7_2;k++)
    //{
      // all_anchors[i][j][k]=shifts[i][k]+ anchor[j][k];
      //}
   //}
//}
//printf(" all anchors value is \n");
//for(int i=0;i<pool_output5 * pool_output5;i++)
//{ 
  //for(int j=0;j<kernel_number_7_1;j++)
  // {
    //for(int k=0;k<kernel_number_7_2;k++)
     //{

   //   printf("  %f  ", all_anchors[i][j][k]);
    //} 
    //  printf(" \n");
//}
//printf("\n");
//}

    // First, select an OpenCL platform to run on.  
     clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);
      std::cout << "number of platform is : " << numPlatforms << std::endl;

     if(!init_opencl()) {
                   return -1;
                    }
              

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platforms[i], 
            CL_DEVICE_TYPE_ALL, 
            0,
            NULL,
            &numDevices);
            
            std::cout << "errNum : " << errNum << std::endl;
            std::cout << "cl_sucess : " << CL_SUCCESS << std::endl;
            std::cout << "cl_device is : " << CL_DEVICE_NOT_FOUND << std::endl;
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {           
                        std::cout << "if part is running " << std::endl;  
			checkErr(errNum, "clGetDeviceIDs");
                       
        }
	    else if (numDevices > 0) 
		{        std::cout << "else if part is running " << std::endl;
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platforms[i],
				CL_DEVICE_TYPE_ALL,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}
          std::cout << "DeviceID is : " << &deviceIDs[1] << std::endl; 
         std::cout << "no else if part " << std::endl;
	// Check to see if we found at least one CPU device, otherwise return
	if (deviceIDs == NULL) {
                 std::cout << "if_device id part is running " << std::endl;
		std::cout << "No FPGA device found" << std::endl;
		exit(-1);
	}
      std::cout << "no if part " << std::endl;

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[i],
        0
    };
    std::cout << "context property is  " << contextProperties <<std::endl;
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
     std::cout << "context is  " << context <<std::endl;
	checkErr(errNum, "clCreateContext");
    std::cout << "maybe error is here after checkERR  " <<std::endl;
	
   std::cout << "maybe error is here after another checkERR  " <<std::endl;
  // added to define program ""'
  std::string binary_file = getBoardBinaryFile("gen_conv", deviceIDs[0]);  ///conv_gen_shif5.cl // conv_gen_shift10_channel
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), deviceIDs, numDevices);
	// Create program from source
   std::cout << "prgram object is   " <<program<<std::endl;
	// Build program
	errNum = clBuildProgram(program, 0, NULL, "", NULL, NULL);
          std::cout << "maybe error is here after this checkERR  " <<std::endl;
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);
    	
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
	    
             checkErr(errNum, "clBuildProgram");
             std::cout << "maybe error is here before another checkERR  " <<std::endl;
    }

	// Create kernel object for r file
	
	// Now allocate buffers
     /////modify window parameters to see performance ///

       window_width = 19;
       window_width_2 = 9;
       window_width_3 = 5;
       window_width_4 = 5;
       window_width_5 = 5; 
  ////////////////////

	inputSignalBuffer_r[0] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width1 * inputSignalHeight_1* inputSignalWidth_1,
		static_cast<void *>(inputSignal_r),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");
     
inputSignalBuffer_r[1] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width1 * inputSignalHeight_1* inputSignalWidth_1,
		static_cast<void *>(inputSignal_r4),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

inputSignalBuffer_r[2] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width1 * inputSignalHeight_1* inputSignalWidth_1,
		static_cast<void *>(inputSignal_r6),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");
for(int d = 0;d<image_num;d++)
{

for(int i = 0; i<227; i++)
    {
      for(int j=0; j<227;j++)
         {
           for(int k =0; k<3;k++)
            {
                  
                 
                inputSignal_r9[k][i][j] = (float) image1[d+3].at<Vec3b>(i,j)[k];  
                
           }
       }      
     } 

for(int i =0; i< input_width_1;i++)
 {
   for(int j =0 ; j< inputSignalHeight_1; j++)
    {
     for(int k =0 ; k< inputSignalWidth_1_1; k++)
      {  
         for(int r =0; r<4;r++)
           {
         
       //    for(int d =0 ; d<image_num;d++)
           {
           inputSignal_r10[j][k][i + r * input_width_1] = inputSignal_r9[i][j][k *4+ r]; 
                     }
          }
     
      }
     }
  }
inputSignalBuffer_r[d+3] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width1 * inputSignalHeight_1* inputSignalWidth_1,
		static_cast<void *>(inputSignal_r10),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

}
         
	maskBuffer_r = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_1 * input_width1* maskheight_1 * maskwidth_1_1,
		static_cast<void *>(conv1_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      maskBuffer_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2 * input_width_2* maskheight_2 * maskwidth_2,
		static_cast<void *>(conv2_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      maskBuffer_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_3 * input_width_3* maskheight_3 * maskwidth_3,
		static_cast<void *>(conv3_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      maskBuffer_4 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_4 * input_width_4* maskheight_4 * maskwidth_4,
		static_cast<void *>(conv4_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
  maskBuffer_5 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_5 * input_width_5* maskheight_5 * maskwidth_5,
		static_cast<void *>(conv5_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
maskBuffer_8 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_8 * masklength_8 * 16,
		static_cast<void *>(fc8_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

maskBuffer_7 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_7 * masklength_7 * 16,
		static_cast<void *>(fc7_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

maskBuffer_6 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  height_6_1 * 16 * 36 * 4 * 16,
		static_cast<void *>(fc6_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
 
bias_Buffer6 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_6,
		static_cast<void *>(fc6_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");    


bias_Buffer7 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_7,
		static_cast<void *>(fc7_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");    

bias_Buffer8 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_8,
		static_cast<void *>(fc8_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");


 
     bias_Buffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_1,
		static_cast<void *>(conv1_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
  bias_Buffer2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_2,
		static_cast<void *>(conv2_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
bias_Buffer3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_3,
		static_cast<void *>(conv3_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
bias_Buffer4 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_4,
		static_cast<void *>(conv4_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
bias_Buffer5 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_5,
		static_cast<void *>(conv5_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
     
output_Buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * conv1_height * conv1_width *input_height_1, /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
outputBuffer_lrn_1 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) *input_height_1* conv1_height* conv1_width , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");


outputBuffer_pool_1 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR  ,
		sizeof(float) * height_pool_1* outputSignalHeight_pool_1 * outputSignalWidth_pool_1  , /// to do padding//// /// pool buffer
 		static_cast<void *>(pool1),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
outputBuffer_conv2 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  ,
		sizeof(float) * input_height_2* conv2_height* conv2_width , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
outputBuffer_lrn_2 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) *input_height_2* conv2_height* conv2_width , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_pool_2 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * height_pool_2 * outputSignalHeight_pool_2 * outputSignalWidth_pool_2 , /// to do padding//// /// pool buffer
 		static_cast<void *>(pool2),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
outputBuffer_conv3 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_3* conv3_height* conv3_width , /// to do padding//// /// pool buffer
 		static_cast<void *>(conv3[0]),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
outputBuffer_conv4 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_height_4* conv4_height* conv4_width , /// to do padding//// /// pool buffer
 		static_cast<void *>(conv4[0]),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_conv5 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * input_height_5* conv5_height* conv5_width , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_pool_3 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) * height_pool_3 * outputSignalHeight_pool_3 * outputSignalWidth_pool_3 , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_fc1 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  ,
		sizeof(float) * input_height_6 , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_fc2 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  ,
		sizeof(float) * input_height_7 , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");

outputBuffer_fc4 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  ,
		sizeof(float) * input_height_7 , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
outputBuffer_fc3 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) * input_height_8 , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");


cl_event event1;
cl_event event2;
cl_event event3;
cl_ulong time_start;
cl_ulong time_end;
cl_ulong time_start1;
cl_ulong time_end1;
int y=0;
int z=1;
double nanosecs[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};

double nanosecs2 = 0 ;
float accuracy = 0;
time_start1 = clock();
int key = 0;
int egf = 0;
inputSignal_r13[0][0][0] = 0;
//image4 =Mat::zeros(227*2,227*2,3);
queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue2 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue3 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue4 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue5 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue6 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue7 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
queue8 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");

for(int dr =0; dr<4;dr++)
{
   
egf = egf + dr * 2;
double nanosecs1 = 0 ;
for(int i=0;i<image_num+3;i++)
{

 Mat image4(cv::Size(227*4,227*3), CV_8UC3, cv::Scalar(0));
std::ostringstream str;  // label
std::ostringstream str1;  // actual time
std::ostringstream str2;   /// frames per sec
std::ostringstream str3;   /// average accuracy
std::ostringstream str4;     /// batch size 
printf(" total exeuction time is   is   %0.3f milisecs\n", nanosecs1/1000000.0);
 nanosecs1 = 0; 
// str << "Here is some text:" << " hey convolution ";
    
  //if( key ==27);
   //break;cd 
 
//// conv1 starts //////////
printf(" conv1 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
int inputSignalWidth2 = 5 * inputSignalWidth;
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputSignalBuffer_r[i]);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv);
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check1);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_1);
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_1);
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv1); // can change for other convolutions ////
//errNum |= clSetKernelArg(kernel1 ,16 , sizeof(cl_uint), &stride_value);
//////printf(" trueeeee");

	// Pick the first device and create command queue.
    const size_t globalWorkSize1[1] = {1};
    const size_t localWorkSize1[1]  = {1};
     const size_t globalWorkSize3[1] = {1};
    const size_t localWorkSize3[1]  = {1}; 
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");




kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputSignalBuffer_r[0]);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv1_height); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_1);
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad); 
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
////printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");


const int fc_param = 32;
const int fc_check1 = !fc_check;

//if(stride ==1)
{
  //outputSignalHeight1 = 2 ;
}
//else
{
//outputSignalHeight1 = outputSignalHeight;
}
const int pad_lrn = 1 ;  // only used before lrn
kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_int), &conv1_width); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_1);
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &conv1_height);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_1);///determines how much to write at when k == window2-1
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &conv1_width);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &conv1_width);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);  // only used before lrn //
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
////printf(" trueeeee");

    const size_t globalWorkSize2[1] = {1};
    const size_t localWorkSize2[1]  = {1};
	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize2, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
 nanosecs [0] = time_end- time_start;
  printf(" execution time for conv1 is   %0.3f milisecs\n", nanosecs[0]/1000000.0);

//clFinish(queue1);
//// lrn1 starts //////////
printf(" lrn1 starts \n");
kernel1 = clCreateKernel(
		program,
		"lrn_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv1_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_lrn_1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_lrn_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window2_lrn_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_lrn_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_lrn_1);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &pad_lrn_1);
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &conv1_height);
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    const size_t globalWorkSize1_lrn[2] = {1 , 1};
    const size_t localWorkSize1_lrn[2]  = {1,1};
     const size_t globalWorkSize3_lrn[1] = {1 };
    const size_t localWorkSize3_lrn[1]  = {1}; 
errNum = clEnqueueNDRangeKernel(
		queue4, 
		kernel1, 
		2, 
		NULL,
        globalWorkSize1_lrn, 
		localWorkSize1_lrn,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"lrn",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv1_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_lrn_1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_lrn_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window2_lrn_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_lrn_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_lrn_1);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &pad_lrn_1);
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &conv1_height);


//printf(" trueeeee");

    const size_t globalWorkSize2_lrn[1] = {1};
    const size_t localWorkSize2_lrn[1]  = {1};
	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue5, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3_lrn, 
		localWorkSize2_lrn,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
  nanosecs[1] = time_end- time_start;
  printf(" execution time for  lrn1 is   %0.3f milisecs\n", nanosecs[1]/1000000.0);

printf(" pool1 starts ////\n");   
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
	checkErr(errNum, "clCreateKernel");
         errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_lrn_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv1_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskWidth_pool_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_1_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_1);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_pool_1); 
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool);
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check_pool);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_pool); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskWidth_pool_1);
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_pool1);  /// will be 64 for fully connected layers otherwise same as maskwidth

	// Pick the first device and create command queue.
 
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event3);
//checkErr(errNum, "clEnqueueNDRangeKernel");

kernel1 = clCreateKernel(
		program,
		"pool",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_lrn_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_1); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &pool_width_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_1_1);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_1);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_pool_1);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue6, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
  nanosecs[2] = time_end- time_start;
  printf(" execution time for  pooling 1  is   %0.3f milisecs\n", nanosecs[2]/1000000.0);
printf(" conv2 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_1);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv2); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_2);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv2_1);  /// will be 64 for fully connected layers otherwise same as maskwidth

//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv2_height); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_2);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");






kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv2_width); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride2);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_2);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &conv2_height);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer2);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_2);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &conv2_width);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &conv2_width);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);

//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[3] = time_end- time_start;
  printf(" execution time for  conv2 is   %0.3f milisecs\n", nanosecs[3]/1000000.0);

///////////// lrn 2 starts ////////
printf(" lrn2 starts \n");
kernel1 = clCreateKernel(
		program,
		"lrn_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv2_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_lrn_2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_lrn_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window2_lrn_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_lrn_2);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_lrn_2);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &pad_lrn_1);  // always fixed to 0//
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &conv2_height);
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    const size_t globalWorkSize1_lrn_2[2] = {1, 1};
    const size_t localWorkSize1_lrn_2[2]  = {1,1};
     const size_t globalWorkSize3_lrn_2[1] = {1 };
    const size_t localWorkSize3_lrn_2[1]  = {1}; 
errNum = clEnqueueNDRangeKernel(
		queue7, 
		kernel1, 
		2, 
		NULL,
        globalWorkSize1_lrn_2, 
		localWorkSize1_lrn_2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"lrn",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv2_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_lrn_2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_lrn_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window2_lrn_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_lrn_2);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_lrn_2);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &pad_lrn_1); //// most of the case it is 0 ////
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &conv2_height);


//printf(" trueeeee");

    const size_t globalWorkSize2_lrn_2[1] = {1};
    const size_t localWorkSize2_lrn_2[1]  = {1};
	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue8, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3_lrn_2, 
		localWorkSize2_lrn_2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[4] = time_end- time_start;
  printf(" execution time for  lrn2 is   %0.3f milisecs\n", nanosecs[4]/1000000.0);

printf(" pool2 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_lrn_2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv2_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskWidth_pool_2);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_2_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_2);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_pool_2); 
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool);/////fixed parameter /////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check_pool_2);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_pool); //// fixed parametr ////
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskWidth_pool_2); 
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_pool1);
	// Pick the first device and create command queue.
 
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event3);
//checkErr(errNum, "clEnqueueNDRangeKernel");

kernel1 = clCreateKernel(
		program,
		"pool",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
   
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_lrn_2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_2);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_2); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &pool_width_2);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_2);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_2);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_2);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_2_1);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_2);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_pool_2);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue6, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[5] = time_end- time_start;
  printf(" execution time for  pooling2 is   %0.3f milisecs\n", nanosecs[5]/1000000.0);

///////// con3 starts ////////
printf(" conv3 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_3);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_2);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_3);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_3);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_3);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_3);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_3);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv3); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check3);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_3); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_3);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv3_1); // can change for other convolutions ////
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_2);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_3);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv3_height); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_3);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_3);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_3);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_3);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_3);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_3);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");




kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer3);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_3);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv3_width); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_3);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_3);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_3);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_3);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_3); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_3);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride3);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_3);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &conv3_height);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer3);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_3);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &conv3_width);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &conv3_width);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[6] = time_end- time_start;
  printf(" execution time for  conv3 is   %0.3f milisecs\n", nanosecs[6]/1000000.0);

printf(" conv4 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv3);     //////outputBuffer_conv4 used for layer 3 somehow outputBuffer_conv3 was not working ///
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_4);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv3_width);   // input feature map width
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_4);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv4);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_4);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_4);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_4);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_4);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv4); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);    //// input feature map height //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check4);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_4); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_4);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv4_1);  /// will be 64 for fully connected layers otherwise same as maskwidth
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv3);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_4);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv3_height); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_4);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv4);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_4);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_4);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_4);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_4);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_4);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer4);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_4);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv4_width); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_4);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv4);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_4);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_4);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_4);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_4); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_4);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride4);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_4);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &conv4_height);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer4);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to //// ///for convolution it is always 4 ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_4);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &conv4_width);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &conv4_width);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[7] = time_end- time_start;
  printf(" execution time for  conv4 is   %0.3f milisecs\n", nanosecs[7]/1000000.0);


/////// conv5 starts //////
printf(" conv5 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv4);     //////outputBuffer_conv4 used for layer 3 somehow outputBuffer_conv3 was not working ///
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_5);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv4_width);   // input feature map width
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_5);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv5);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_5);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_5);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_5);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_5);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv5); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);    //// input feature map height //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check5);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_5); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_5);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv5_1);  /// will be 64 for fully connected layers otherwise same as maskwidth
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
   

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv4);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_5);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv4_height); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_5);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv5);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_5);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_5);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_5);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_5);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_5);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer5);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_5);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv5_width); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_5);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_conv5);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_5);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_5);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_5);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_5); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_5);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride5);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_5);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &conv5_height);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer5);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to //// ///for convolution it is always 4 ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_5);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &conv5_width);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &conv5_width);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[8] = time_end- time_start;
  printf(" execution time for  conv5 is   %0.3f milisecs\n", nanosecs[8]/1000000.0);
printf("  pooling 3  \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
	checkErr(errNum, "clCreateKernel");
        


  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv5);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_5);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &conv5_width);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskWidth_pool_3);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_3);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_3);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_3_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_3);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_pool_3); 
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool);/////fixed parameter /////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check_pool_3);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_pool); //// fixed parametr ////
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskWidth_pool_3);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_pool3);
	// Pick the first device and create command queue.
 
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event3);
//checkErr(errNum, "clEnqueueNDRangeKernel");

kernel1 = clCreateKernel(
		program,
		"pool",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_conv5);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_5);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_3); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &pool_width_3);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_pool_3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_pool_3);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_pool_3);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_pool_3_1);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_pool_3);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_pool_3);
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue6, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[9] = time_end- time_start;
  printf(" execution time for  pooling 5  is   %0.3f milisecs\n", nanosecs[9]/1000000.0);

printf(" fully connected layer 1  starts \n");
//////////////////////////////////
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       


  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_3);     //////outputBuffer_conv4 used for layer 3 somehow outputBuffer_conv3 was not working ///
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_6);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv);   // already defined /// need to be 32
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_6);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_6);      
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_6);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_6);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_6);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv6); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);    //// input feature map height //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check6);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_6); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_6);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv1); 
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_3);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_6);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_6);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_6);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_6);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_6);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_6);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_6);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask_fc); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2_fc);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3_fc);   
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");




kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer6);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_6);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_int), &outputSignalHeight6); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_6);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc1);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_6);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_6);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_6);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_6); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_6);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride6);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_6);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &outputSignalHeight6);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer6);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to //// ///for convolution it is always 4 ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_6);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &outputSignalHeight6);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &outputSignalHeight6);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[10] = time_end- time_start;
  printf(" execution time for fully connected layer 1  is   %0.3f milisecs\n", nanosecs[10]/1000000.0);


///fully connected layer 2 starts /////
printf(" fully connected layer 2 starts \n");
 
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       


  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_fc1);     //////outputBuffer_conv4 used for layer 3 somehow outputBuffer_conv3 was not working ///
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_7);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv);   // already defined /// need to be 32
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_7);  /// maskwidth_1 acting as window for mask16 
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc4); 
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_7);  
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_7);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_7);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_7);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv7); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);    //// input feature map height //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check7);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_7); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_7);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv1); 
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_fc1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_7);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_7);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc4);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_7);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_7);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_7);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_7);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_7);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask_fc); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2_fc);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3_fc);   //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer7);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_7);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_int), &outputSignalHeight7); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_7);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc4);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_7);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_7);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_7);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_7); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_7);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride7);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_7);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &outputSignalHeight7);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer7);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to //// ///for convolution it is always 4 ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_7);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &outputSignalHeight7);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &outputSignalHeight7);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[11] = time_end- time_start;
  printf(" execution time for fully connected 2  is   %0.3f milisecs\n", nanosecs[11]/1000000.0);
/////// fully connected layer 3 started  //////
printf(" fully connected layer 3 started \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_fc4);     //////outputBuffer_conv4 used for layer 3 somehow outputBuffer_conv3 was not working ///
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_8);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv);   // already defined /// need to be 32
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_8);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_8);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_8);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_8);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_8);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_conv8); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);    //// input feature map height //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check8);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_8); /// will be 0 for non grouped convolution  only 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_8);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_conv1);
//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");

printf(" true \n");


kernel1 = clCreateKernel(
		program,
		"mask_read",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
      

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_fc4);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_8);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth_conv); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_8);  /// maskwidth_2 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_8);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_8);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_8);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_8);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_8);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask_fc); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2_fc);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3_fc);   //
//errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_mem), &maskBuffer_8_1);   //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue2, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");


kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
       

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bias_Buffer8);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_8);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalHeight8); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_8);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &outputBuffer_fc3);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_8);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_8);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_8);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_8); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_8);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride8);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_8);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &outputSignalHeight8);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &bias_Buffer8);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to //// ///for convolution it is always 4 ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_8);  ///only need ot be changed if there is padding in next layer ////
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &outputSignalHeight8);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &outputSignalHeight8);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param);
//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize3, 
		localWorkSize2,
        0, 
		NULL, 
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");
clWaitForEvents(1, &event1 );
clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);

  nanosecs[12] = time_end- time_start;
  printf(" execution time for fully connected 3  is   %0.3f milisecs\n", nanosecs[12]/1000000.0);

errNum = clEnqueueReadBuffer(
		queue3, 
		outputBuffer_fc3, 
		CL_TRUE,
        0, 
		sizeof(float) * input_height_8  ,  ///conv1_height * conv1_width *input_height_1  ,  height_pool_1* outputSignalHeight_pool_1 * outputSignalWidth_pool_1
		fc_8[0],
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
//printf("total execution time is \n");
//putText(image1[i], " detectrion ");
time_end1 = clock();

 
 //clFinish(queue3);
  //clWaitForEvents(1,&event3);

    //printf(" cpu execution time is %f \n", cpu_time_used);
printf(" true \n");

std::cout<<" output result is   "<<std::endl;
//printf(" conv1_width is 
int r=0;
//for(int ef =0 ;ef<3;ef++)
//{
float max[5] = {0,0,0,0,0};
int index[5] = {0,0,0,0,0};
for(int ef =0;ef<1000;ef++)
{

if( fc_8[0][ef] > max[0])
  {
       //printf(" value of index and data is  %d %f\n", i, fc_8[0][i]);
       for(int j = 4;j>0;j--)
         {
             max[j] = max[j-1];
             index[j] = index[j-1];
          }
        max[0] = fc_8[0][ef];
        index[0] = ef;
   }

}
nanosecs1 = 0;
for(int eg =0; eg<13;eg++)
{
   nanosecs1 += nanosecs[eg];
}
nanosecs2 += nanosecs1;
int tmp =0;
if(index[0] == label1[i])
{
      tmp = 1;
}
else
{
    tmp  = 0;
}
accuracy += tmp;

str << " Image  is " << label[index[0]];
str1 << " time is " << nanosecs1/1000000.0;
str2 << " Frame per secs  is " << (1/(nanosecs2/(1000000.0*(dr *14 + i+1)))) * 1000;
str3 << "  Accuracy percentage is  " << (accuracy/(dr * 14 + i+1)) * 100 ;
str4 << "Batch Size is " << 1  ;
int j = i * 100 + 5;
cv::putText(image4, str.str(),  cv::Point(227, 15  ), FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,1,0);
cv::putText(image4, str1.str(),  cv::Point(227,45 ), FONT_HERSHEY_PLAIN,1, Scalar(0,255,0),1,8,0);
cv::putText(image4, str2.str(),  cv::Point(227, 65 ), FONT_HERSHEY_PLAIN,1, Scalar(200,200,180),1,8,0);
cv::putText(image4, str3.str(),  cv::Point(227,85 ), FONT_HERSHEY_PLAIN,1, Scalar(255,143,143),1,8,0);
cv::putText(image4, str4.str(),  cv::Point(227, 105 ), FONT_HERSHEY_PLAIN,1, Scalar(255,143,143),1,8,0);
Mat roi = image4(Rect(0,0,227,227));
image1[i].copyTo(roi);
imshow(" Alexnet_model", image4);

//if( i == egf)
   {
   waitKey(2);
    }
 	if( i ==image_num+2 && dr ==3 )
    key = waitKey(0);
//printf("top5 maximums are \n");
for(int eg =0;eg<1;eg++)
{
   printf(" image is  %s %d %f \n", label[index[eg]], index[eg], max[eg]);
}
}
}
//printf(" value fo fc8 is %f\n", fc_8[0][703]);
// Output the result buffer//
//for(int w=0;w<1;w++)
{
//for (int z=0; z<16 ;z++)
{
// printf(" value of z is %d\n", z);
//for (int y = 0; y < 3 ; y++)
	{
 //         printf(" value o f y is %d\n", y);
  //	for (int x = 0; x <outputSignalWidth_pool_3 ;x++)
		{
                    //   std::cout << x+y*(pool_output7)+ z*(pool_output7)*(pool_output7) +w << "        ";  
// std::cout << pool3[0][0][y][x][z]<< " ";   //conv2[0][y][x][z]  ///  pool1[0 + y * outputSignalWidth_pool_1  * 16 + x * 16 + pad_pool_1+ z]15
  
		}
// std::cout << std::endl;
	}
	std::cout << std::endl;
}
std::cout<< "one ";
}
printf(" first 80 is \n");
//for(int i =0 ;i< 16;i++)
{
//  printf(" %f  ", pool1[1][2][0][i]);
}
//printf(" execution time is  %0.3f milisecs", nanosecs/1000000.0);
     //cleanup();


	return 0;
}
/////functions added later /////////
// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("INTEL");
  std::cout << std::endl << "platform is " << platform << std::endl;
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
        }
  else {
   printf("Platform: %s\n", getPlatformName(platform).c_str());
   return true;
}
}
//////clean up part added later /////////






 


