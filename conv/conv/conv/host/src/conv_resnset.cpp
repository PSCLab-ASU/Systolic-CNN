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
//#include "img9.h"   // park_bench.jpg
#include "img6.h"
//#include "img9_res.h"
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
//#include "img8.h"
#include "conv1_w.h"
#include "pool_params.h"
#include "res_2_0_a_w.h"
#include "res_2_0_b_w.h"
#include "res_2_1_w.h"
#include "lrn_params.h"
#include "res_2_0_c_w.h"
#include "res_2_1_a_w.h"
#include "params.h"
#include "res_2_1_b_w.h"
#include "res_2_1_c_w.h"
#include "res_2_2_a_w.h"
#include "res_2_2_b_w.h"
#include "res_2_2_c_w.h"
#include "res_3_0_a_w.h"
#include "res_3_0_b_w.h"
#include "reoder_weight.h"
#include "reoder_weight1.h"
#include "reoder_data.h"
#include "res_3_1_w.h"
#include "res_3_0_c_w.h"
#include "res_3_1_a_w.h"
#include "res_3_1_b_w.h"
#include "res_3_1_c_w.h"
#include "res_3_2_a_w.h"
#include "res_3_2_b_w.h"
#include "res_3_2_c_w.h"
#include "res_3_3_a_w.h"
#include "res_3_3_b_w.h"
#include "res_3_3_c_w.h"
#include "res_4_0_a_w.h"
#include "res_4_0_b_w.h"
#include "res_4_0_1_w.h"
#include "res_4_0_c_w.h"
#include "res_4_1_a_w.h"
#include "res_4_1_b_w.h"
#include "res_4_1_c_w.h"
#include "res_4_2_a_w.h"
#include "res_4_2_b_w.h"
#include "res_4_2_c_w.h"
#include "res_4_3_a_w.h"
#include "res_4_3_b_w.h"
#include "res_4_3_c_w.h"
#include "res_4_4_a_w.h"
#include "res_4_4_b_w.h"
#include "res_4_4_c_w.h"
#include "res_4_5_a_w.h"
#include "res_4_5_b_w.h"
#include "res_4_5_c_w.h"
#include "res_5_0_a_w.h"
#include "res_5_0_b_w.h"
#include "res_5_0_1_w.h"
#include "res_5_0_c_w.h"
#include "res_5_1_a_w.h"
#include "res_5_1_b_w.h"
#include "res_5_1_c_w.h"
#include "res_5_2_a_w.h"
#include "res_5_2_b_w.h"
#include "res_5_2_c_w.h"
#include "res_pool_w.h"
#include "pool_data.h"
#include "res_fc_w.h"
//#include "conv3_w.h"
//#include "conv4_w.h"
//#include "conv5_w.h"
//#include "fc6.h"
//#include "fc7.h"
//#include "fc8.h"
//#include <python2.6/Python.h>
#include "label.h"
#include "label_check.h"

#define num_image 12
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
	cl_program program;
	cl_kernel kernel1;
        cl_kernel kernel2;
        cl_kernel kernel3;  
	cl_mem inputSignalBuffer_r[num_image];
        cl_mem inputSignalBuffer_g;
        cl_mem inputSignalBuffer_b;
        cl_mem pool1Buffer;
        cl_mem res2_0_branch1_bnBuffer;
	cl_mem maskBuffer_r;
        cl_mem maskBuffer_g;
        cl_mem maskBuffer_b;
        cl_mem bias_Buffer;
        cl_mem output_Buffer;
        cl_mem outputBuffer_pool_1;
        cl_mem maskBuffer_res2_0_a;
        cl_mem biasBuffer_res2_0_a;
        cl_mem res2_0_a_buffer;
        cl_mem maskBuffer_res2_0_b;
        cl_mem biasBuffer_res2_0_b;
        cl_mem res2_0_b_buffer; 
        cl_mem maskBuffer_res2_0_1;
        cl_mem biasBuffer_res2_0_1;
        cl_mem res2_0_1_buffer;
        cl_mem res2_1_1_buffer;
        cl_mem maskBuffer_res2_0_c;
        cl_mem biasBuffer_res2_0_c;
        cl_mem resbuffer[conv_layers+1]; 
        cl_mem resbuffer1;
        cl_mem resbuffer2;
        cl_mem maskBuffer_res[conv_layers];
        cl_mem biasBuffer_res[conv_layers];
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
const int stride_write = 0;   
const int stride_write1 = 1;  // not needed currently ///
const int sumadd_no = 1;
const int sumadd_no1 = 0;
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

for(int i = 1;i< conv_layers;i++)
{
   height_res[i] = input_height_res[i] / 16;
   kernel_width_res[i] =  input_width_res[i]/16;
   stride_param[i] = stride_fact_res[i];
   window_check2_res[i] = input_height_res[i];
}
height_res[8] = 1;
window_check2_res[conv_layers-2 ] = -1;
pool1_2_res[conv_layers-2] = 1;

//// reorder input //////////////
Mat image[num_image], image1[num_image], image2, image3;
image[0] = imread("image_location/park_bench.jpg");
image[1] = imread("image_location/flower.jpg");
image[2] = imread("image_location/pizza.jpg");
image[3] = imread("image_location/violin.jpg");
image[4] = imread("image_location/frog.jpg");
image[5] = imread("image_location/car.jpg");
image[6] = imread("image_location/sumglass.jpg");
image[7] = imread("image_location/terrier.jpg");
image[8] = imread("image_location/snake.jpg");
image[9] = imread("image_location/matchstick.jpg");
image[10] = imread("image_locationespresso.jpg");
image[11] = imread("image_location/mouse.jpg");
if(image[0].empty() || image[1].empty() || image[2].empty() || image[3].empty() || image[4].empty(), image[5].empty())
{
  printf(" error in loading image ");
}
else{
namedWindow( "Display Image", 1000);
for(int i =0; i<num_image;i++)
{
resize(image[i],image1[i],Size(224,224),0,0);
}
//resize(image3,image1[2],Size(224,224),0,0);
//resize(image3,image1[2],Size(227,227),0,0);
//image1.convertTo(image2, CV_32FC3);
 //cout << (float) image1.at<Vec3b>(0,1)[0] << "" "" << endl;
  for(int i = 0; i<224; i++)
    {
      for(int j=0; j<224;j++)
         {
           //for(int k =0; k<3;k++)
            {
                 inputSignal_r1[0][i][j] = ((float) image1[0].at<Vec3b>(i,j)[0]-123)/58.395;
                 inputSignal_r1[1][i][j] = ((float) image1[0].at<Vec3b>(i,j)[1]-117)/57.12;
                 inputSignal_r1[2][i][j] = ((float) image1[0].at<Vec3b>(i,j)[2]-104)/57.375; 
          }
       }      
     }    

 printf("true \n");

}

printf(" value of image is %f %f %f %f %f %f %f %f %f %f %f \n", inputSignal_r1[0][0][0],inputSignal_r1[0][0][1], inputSignal_r1[0][0][2],inputSignal_r1[1][0][0],inputSignal_r1[1][0][1], inputSignal_r1[1][0][2],inputSignal_r1[2][0][0],inputSignal_r1[2][0][1], inputSignal_r1[2][0][2],(float) image1[0].at<Vec3b>(0,0)[1], (float) image1[0].at<Vec3b>(0,0)[2]);
{
for(int i =0; i<input_width_1;i++)
 {
   for(int j =0; j<inputSignalHeight_1;j++)
    {
      for(int k =0 ;k< inputSignalWidth_1;k++)
       {
         inputSignal_r[i][j+pad1][k+pad1] = inputSignal_r1[i][j][k];  
        }
      }
    }
for(int i =0; i<input_width_1;i++)
 {
   for(int j =0; j<inputSignalHeight_1;j++)
    {
      for(int w =0 ;w< (inputSignalWidth_1 + 2 * pad1+10)/12;w++) /////// (inputSignalWidth_1 + 2 * pad1+10) this is done to make it divisible by 12 /////
       {
        for(int k = 0 ;k<2;k++)
          {
         for(int l =0 ; l< 16;l++)
           {
           // inputSignal_r2[i][j][w * 16 * 2+ k* 16 + l ] = inputSignal_r[i][j][ w* 12 + k * 2  + l ]; 
           }
          } 
        }
      }
    }


 }


//else
{
 
for(int i =0; i< input_width_1;i++)
 {
   for(int j =0 ; j< (inputSignalHeight_1+2 * pad1); j++)
    {
     for(int k =0 ; k< (inputSignalWidth_1+2 * pad1) ; k++)
      {  
        // for(int r =0; r<4;r++)
           {
         inputSignal_r4[j][k][i] = inputSignal_r[i][j][k]; 
         
          }
     
      }
     }
  }
 }
printf(" input value is of convolution is  %f \n", inputSignal_r4[3][3][0]);
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
        //   for(int z =0 ; z<4;z++)
             {
          
            conv1_w_1[y][k][r][i][j] = conv1_w[y * ll/4 +  i][j][k][r];
            //if( r == maskwidth_1_1 -1 && z ==2)
              {
              //  z++;  /// need to be done for only 11x11 convolution 
              } 
         }
        }
     }
   }
  }
  }
 }
for(int y =0;y<height_2_a * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2_a; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2_a;k++)
      {
        for(int r =0;r<maskwidth_2_a;r++)
         {
       res2_a_0_w_1[y][z][k][r][i][j] = res2_a_0_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_2_b * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2_b; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2_b;k++)
      {
        for(int r =0;r<maskwidth_2_b;r++)
         {
       res2_b_0_w_1[y][z][k][r][i][j] = res2_b_0_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_2_c * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2_c; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2_c;k++)
      {
        for(int r =0;r<maskwidth_2_c;r++)
         {
       res2_c_0_w_1[y][z][k][r][i][j] = res2_c_0_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_2_1 * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2_1; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2_1;k++)
      {
        for(int r =0;r<maskwidth_2_1;r++)
         {
       res2_1_0_w_1[y][z][k][r][i][j] = res2_1_0_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_2_1_a * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_2_1_a; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_2_1_a;k++)
      {
        for(int r =0;r<maskwidth_2_1_a;r++)
         {
       res2_1_a_w_1[y][z][k][r][i][j] = res2_1_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}

for(int y =0;y<height_res[1] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[1]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[1];k++)
      {
        for(int r =0;r<maskwidth_res[1];r++)
         {
       res2_1_b_w_1[y][z][k][r][i][j] = res2_1_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[2] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[2]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[2];k++)
      {
        for(int r =0;r<maskwidth_res[2];r++)
         {
       res2_1_c_w_1[y][z][k][r][i][j] = res2_1_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[3] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[3]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[3];k++)
      {
        for(int r =0;r<maskwidth_res[3];r++)
         {
       res2_2_a_w_1[y][z][k][r][i][j] = res2_2_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}

for(int y =0;y<height_res[4] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[4]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[4];k++)
      {
        for(int r =0;r<maskwidth_res[4];r++)
         {
       res2_2_b_w_1[y][z][k][r][i][j] = res2_2_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}

for(int y =0;y<height_res[5] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[5]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[5];k++)
      {
        for(int r =0;r<maskwidth_res[5];r++)
         {
       res2_2_c_w_1[y][z][k][r][i][j] = res2_2_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[6] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[6]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[6];k++)
      {
        for(int r =0;r<maskwidth_res[6];r++)
         {
       res3_0_a_w_1[y][z][k][r][i][j] = res3_0_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[7] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[7]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[7];k++)
      {
        for(int r =0;r<maskwidth_res[7];r++)
         {
       res3_0_b_w_1[y][z][k][r][i][j] = res3_0_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[9] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[9]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[9];k++)
      {
        for(int r =0;r<maskwidth_res[9];r++)
         {
       res3_1_w_1[y][z][k][r][i][j] = res3_1_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int i =0 ; i <2;i++)
 {
   for(int j =0 ;j< 16;j++)
    {
     for(int r = 0 ; r< res_height_8;r++)
       {
         for(int z =0; z< res_width_8;z++)
           {
             for(int x =0 ; x<16;x++)
               {
                   res_reoder_1[i][j][r][z][x] = 0;
            }
        }
     }
  }
 }

for(int i =0 ; i <2;i++)
 {
   for(int j =0 ;j< 28;j++)
    {
     for(int r = 0 ; r< 10;r++)
       {
         for(int z =0; z< 32;z++)
           {
            for(int r =0; r<3;r++)
             {
             for(int x =0 ; x<16;x++)
               {
                   res_reoder_2[i][j][r][z][r][x] = 1;
            }
          }
        }
     }
  }
 }

for(int y =0;y<height_res[10] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[10]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[10];k++)
      {
        for(int r =0;r<maskwidth_res[10];r++)
         {
       res3_0_c_w_1[y][z][k][r][i][j] = res3_0_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
for(int y =0;y<height_res[12] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[12]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[12];k++)
      {
        for(int r =0;r<maskwidth_res[12];r++)
         {
       res3_1_a_w_1[y][z][k][r][i][j] = res3_1_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
int ef = 13;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_1_b_w_1[y][z][k][r][i][j] = res3_1_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_1_c_w_1[y][z][k][r][i][j] = res3_1_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_2_a_w_1[y][z][k][r][i][j] = res3_2_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_2_b_w_1[y][z][k][r][i][j] = res3_2_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
 printf ( " value od input heihg ti s %d %d %d %d %d \n", ef, input_height_res[ef], input_width_res[ef], maskheight_res[ef],maskwidth_res[ef]);
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_2_c_w_1[y][z][k][r][i][j] = res3_2_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}

ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_3_a_w_1[y][z][k][r][i][j] = res3_3_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_3_b_w_1[y][z][k][r][i][j] = res3_3_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res3_3_c_w_1[y][z][k][r][i][j] = res3_3_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_0_a_w_1[y][z][k][r][i][j] = res4_0_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_0_b_w_1[y][z][k][r][i][j] = res4_0_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+2;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_0_1_w_1[y][z][k][r][i][j] = res4_0_1_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_0_c_w_1[y][z][k][r][i][j] = res4_0_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_1_a_w_1[y][z][k][r][i][j] = res4_1_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_1_b_w_1[y][z][k][r][i][j] = res4_1_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_1_c_w_1[y][z][k][r][i][j] = res4_1_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_2_a_w_1[y][z][k][r][i][j] = res4_2_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_2_b_w_1[y][z][k][r][i][j] = res4_2_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_2_c_w_1[y][z][k][r][i][j] = res4_2_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_3_a_w_1[y][z][k][r][i][j] = res4_3_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_3_b_w_1[y][z][k][r][i][j] = res4_3_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_3_c_w_1[y][z][k][r][i][j] = res4_3_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_4_a_w_1[y][z][k][r][i][j] = res4_4_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_4_b_w_1[y][z][k][r][i][j] = res4_4_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_4_c_w_1[y][z][k][r][i][j] = res4_4_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_5_a_w_1[y][z][k][r][i][j] = res4_5_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_5_b_w_1[y][z][k][r][i][j] = res4_5_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res4_5_c_w_1[y][z][k][r][i][j] = res4_5_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_0_a_w_1[y][z][k][r][i][j] = res5_0_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_0_b_w_1[y][z][k][r][i][j] = res5_0_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_0_1_w_1[y][z][k][r][i][j] = res5_0_1_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
printf(" value of d is %d %d %d %d %d %d \n", ef,height_res[ef], kernel_width_res[ef],maskheight_res[ef],maskwidth_res[ef],kernel_width_res_1[ef]);
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_0_c_w_1[y][z][k][r][i][j] = res5_0_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_1_a_w_1[y][z][k][r][i][j] = res5_1_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_1_b_w_1[y][z][k][r][i][j] = res5_1_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_1_c_w_1[y][z][k][r][i][j] = res5_1_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_2_a_w_1[y][z][k][r][i][j] = res5_2_a_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_2_b_w_1[y][z][k][r][i][j] = res5_2_b_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res5_2_c_w_1[y][z][k][r][i][j] = res5_2_c_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< kernel_width_res[ef]; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res_pool_w_1[y][z][k][r][i][j] = res_pool_w[y * ll/4 + i][z * 16 + j][k][r];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
printf(" data is %d %d %d ", height_res[ef], maskheight_res[ef],maskwidth_res[ef]);
for(int y =0;y<height_res[ef] * 4 ;y++)           // used to be height but now it should be height/2
{
for(int w =0; w<8;w++)
    {
for(int i=0;i<4;i++)  // replaces instead of ll 
 {
   for(int z =0; z< masklength/16; z++)
     {
  for(int j =0; j< 16;j++)
   {
    for(int k =0;k< maskheight_res[ef];k++)
      {
        for(int r =0;r<maskwidth_res[ef];r++)
         {
       res_fc_w_1[y][z][i][j] = res_fc_w[y * ll/4 + i][z * 16 + j];
         }
     }
   }
  }
  }
 }
}
ef = ef+1;
printf(" value fo weight si %f %f \n", res3_2_c_w_1[8][0][0][0][0][0], res3_2_c_w[8][0][0][0]);
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


printf(" value of input is %d %d %d  %f %f \n", input_width_1, input_width,  input_width1,  inputSignal_r[0][12][0], inputSignal_r[0][0][216]);
printf(" value of conv is %d, %f %f %f  \n",height/2,  conv1_w_1[0][0][0][4][0],conv1_w[0][0][0][0], conv1_w[0][0][0][1]);

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
       

     for(int d = 0; d<num_image;d++)
     {
 for(int i = 0; i<224; i++)
    {
      for(int j=0; j<224;j++)
         {
           //for(int k =0; k<3;k++)
            {
                 inputSignal_r1[0][i][j] = ((float) image1[d].at<Vec3b>(i,j)[0]-123)/58.395;
                 inputSignal_r1[1][i][j] = ((float) image1[d].at<Vec3b>(i,j)[1]-117)/57.12;
                 inputSignal_r1[2][i][j] = ((float) image1[d].at<Vec3b>(i,j)[2]-104)/57.375; 
          }
       }      
     }    
 
        for(int i =0; i<input_width_1;i++)
       {
   for(int j =0; j<inputSignalHeight_1;j++)
    {
      for(int k =0 ;k< inputSignalWidth_1;k++)
       {
         inputSignal_r[i][j+pad1][k+pad1] = inputSignal_r1[i][j][k];  
        }
      }
    }
     for(int i =0; i< input_width_1;i++)
 {
   for(int j =0 ; j< (inputSignalHeight_1+2 * pad1); j++)
    {
     for(int k =0 ; k< (inputSignalWidth_1+2 * pad1) ; k++)
      {  
        // for(int r =0; r<4;r++)
           {
         inputSignal_r4[j][k][i] = inputSignal_r[i][j][k]; 
         
          }
     
      }
     }
  }

	inputSignalBuffer_r[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width1 * (inputSignalHeight_1 + 2 * pad1)* (inputSignalWidth_1+ 2 * pad1),
		static_cast<void *>(inputSignal_r4),
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
    
        maskBuffer_res2_0_a = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_a * input_width_2_a* maskheight_2_a * maskwidth_2_a,
		static_cast<void *>(res2_a_0_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

       maskBuffer_res2_0_b = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_b * input_width_2_b* maskheight_2_b * maskwidth_2_b,
		static_cast<void *>(res2_b_0_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      
       maskBuffer_res2_0_c = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_c * input_width_2_c* maskheight_2_c * maskwidth_2_c,
		static_cast<void *>(res2_c_0_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

        maskBuffer_res2_0_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_1 * input_width_2_1* maskheight_2_1 * maskwidth_2_1,
		static_cast<void *>(res2_1_0_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
       int j = 0;
           maskBuffer_res[0] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_1_a * input_width_2_1_a* maskheight_2_1_a * maskwidth_2_1_a,
		static_cast<void *>(res2_1_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
        j = j+1;

       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res2_1_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res2_1_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res2_2_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res2_2_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res2_2_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_0_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_0_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   256,
		static_cast<void *>(reoder_w),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_1_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_0_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
      maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   256,
		static_cast<void *>(reoder_w1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_1_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_1_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_1_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_2_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_2_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
 printf(" value fo j is %d\n", j);
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_2_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_3_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_3_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res3_3_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_0_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_0_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   256,
		static_cast<void *>(reoder_w),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_0_1_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_0_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_1_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_1_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_1_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_2_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_2_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_2_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_3_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_3_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_3_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_4_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_4_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_4_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_5_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_5_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res4_5_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_0_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_0_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_0_1_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_0_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_1_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_1_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_1_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_2_a_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_2_b_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res5_2_c_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res_pool_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
j = j+1;
       maskBuffer_res[j] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_res[j] * input_width_res[j]* maskheight_res[j] * maskwidth_res[j],
		static_cast<void *>(res_fc_w_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");


     bias_Buffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_1,
		static_cast<void *>(conv1_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");


      biasBuffer_res2_0_a = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_a,
		static_cast<void *>(res2_a_0_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      biasBuffer_res2_0_b = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_b,
		static_cast<void *>(res2_b_0_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)"); 
     biasBuffer_res2_0_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_1,
		static_cast<void *>(res2_1_0_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)"); 
    biasBuffer_res2_0_c = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height_2_c,
		static_cast<void *>(res2_c_0_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
int d =0;
     biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_1_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
      d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_1_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
   d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_1_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
 d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_2_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_2_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res2_2_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_0_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_0_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   256,
		static_cast<void *>(res_reorder_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_1_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_0_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;

biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   16,
		static_cast<void *>(res_reorder_b_1),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_1_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_1_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_1_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_2_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_2_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;

biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_2_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;

biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_3_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_3_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res3_3_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_0_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_0_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   256,
		static_cast<void *>(res_reorder_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_0_1_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;

biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_0_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_1_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_1_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_1_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_2_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_2_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_2_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_3_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_3_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_3_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_4_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_4_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_4_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_5_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_5_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res4_5_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_0_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_0_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_0_1_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_0_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_1_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_1_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_1_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_2_a_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_2_b_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res5_2_c_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res_pool_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
d = d+1;
biasBuffer_res[d] = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *   input_height_res[d],
		static_cast<void *>(res_fc_b),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");

output_Buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR  ,
		sizeof(float) * conv1_height * conv1_width *input_height_1, /// to do padding////
 		static_cast<void *>(conv1),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
outputBuffer_pool_1 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * height_pool_1* outputSignalHeight_pool_1 * outputSignalWidth_pool_1  , /// to do padding//// /// pool buffer
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output1)");
res2_0_a_buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_2_a * res2_width_a * res2_height_a ,  /// to do padding////
 		static_cast<void *>(res2_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
res2_0_b_buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * input_height_2_b * (res2_width_b+1)/3 * res2_height_b * 3,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
res2_0_1_buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * input_height_2_1 * res2_width_1 * res2_height_1 ,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
res2_1_1_buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) *  input_height_2_1 * res2_width_1_4 * res2_height_1 * 3 ,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
resbuffer[0]= clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * input_height_2_c * res2_width_c_4 * res2_height_c* 3 ,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");

for(int i =0 ;i<conv_layers;i++)
{
resbuffer[i+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY   ,
		sizeof(float) * input_height_res[i] * res_height[i]* res_width_4[i] ,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");

}
printf(" data is %d %d %d ", input_height_res[conv_layers-1], res_height[conv_layers-1],res_width_4[conv_layers-1]);
 resbuffer[9] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) * 256 * res_height[8]* res_width_4[8] * 2 ,  /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
int g =  0;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR  ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res2_1_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 3;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res2_2_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 6;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res3_0_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 12;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res3_1_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 15;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res3_2_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 18;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res3_3_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 21;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_0_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 26;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_1_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 29;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_2_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 32;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_3_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 35;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_4_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 38;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res4_5_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 41;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res5_0_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 45;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res5_1_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
g = 48;
resbuffer[g+1] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[g] * res_height[g]* res_width[g] ,  /// to do padding////
 		static_cast<void *>(res5_2_a),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");

resbuffer[10] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * input_height_res[9] * res_height[9]* res_width_4[9] ,  /// to do padding////
 		static_cast<void *>(res3_1),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
resbuffer1 = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR   ,
		sizeof(float) * 2 * res2_height_c * res2_width_c_4 * height_2_c * 3 *ll,  /// to do padding////
 		static_cast<void *>(fc_data),
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
resbuffer[12] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR    ,
		sizeof(float) * 2 * 30 *10 * 32 * 3 * 16  ,  /// to do padding////
 		static_cast<void *>( res_reoder_2),  
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
resbuffer[24] = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY  | CL_MEM_COPY_HOST_PTR    ,
		sizeof(float) * 2 * 30 * 30 * 512,  /// to do padding////
 		static_cast<void *>( res_reoder_3),  
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
 

cl_event event1;
cl_event event2;
cl_event event3;
cl_ulong time_start;
cl_ulong time_end;
cl_ulong time_start1;
cl_ulong time_end1;
int y=0;
int z=1;
int egf = 0;
double nanosecs2 = 0 ;
float accuracy = 0;
double nanosecs[conv_layers + 6 ] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
time_start1 = clock();
int key = 0;
for(int xy =0 ;xy<4;xy++)
{
for(int df=0;df<num_image;df++)
{

std::ostringstream str;  // label
std::ostringstream str1;  // actual time
std::ostringstream str2;   /// frames per sec
std::ostringstream str3;   /// average accuracy
std::ostringstream str4;     /// batch size 
//waitKey(50);
//// conv1 starts //////////
printf(" conv1 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");
int inputSignalWidth2 = 5 * inputSignalWidth;
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputSignalBuffer_r[df]);
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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputSignalBuffer_r[i]);
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
		queue1, 
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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem),  &bias_Buffer);
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
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_1);  // only used before lrn //
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sumadd_no);
////printf(" trueeeee");

    const size_t globalWorkSize2[1] = {1};
    const size_t localWorkSize2[1]  = {1};
	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue1, 
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
//// pool1 starts //////////
printf(" pool1 starts ////\n");   
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &output_Buffer);
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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &output_Buffer);
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
		queue1, 
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
nanosecs [1] = time_end- time_start;
printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[1]/1000000.0);


//// res2_0_a starts///////
printf(" res2_0_a starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_a);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_1);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_a);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_a_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_a);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_a);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_a);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_a);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_res2_a); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2_a);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2_a); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_2_a);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_res2_a_1);  /// will be 64 for fully connected layers otherwise same as maskwidth

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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_a);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_height_a); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_a);  /// maskwidth_2_a acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_a_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_a);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_a);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_a);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_a);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_2_a);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

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
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &biasBuffer_res2_0_a );
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_a);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_a); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_a);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_a_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_a);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_a);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_a);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_a); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2_a);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride2_a);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_2_a);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &res2_height_a);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &biasBuffer_res2_0_a);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_2_a);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &res2_width_a);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &res2_width_a);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_2_a);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sumadd_no);

//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue1, 
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
nanosecs [2] = time_end- time_start;
  printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[2]/1000000.0);



////// res2_0_b starts /////
printf(" res2_0_b starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &res2_0_a_buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_b);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_a);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_b);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_b_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_b);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_b);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_b);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_b);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_res2_b); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2_b);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2_b); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_2_b);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_res2_1_b);  /// will be 64 for fully connected layers otherwise same as maskwidth

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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &res2_0_a_buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_b);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_height_b); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_b);  /// maskwidth_2_b acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_b_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_b);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_b);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_b);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_b);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_2_b);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

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
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &biasBuffer_res2_0_b );
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_b);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_b_1); /// includes padding also   // inputwidth1
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_b);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_0_b_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_b);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_b);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_b);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_b); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2_b);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride2_b);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_2_b);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &res2_height_b_1);       /// outputdim
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &biasBuffer_res2_0_b);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_2_b);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &res2_width_b_2);   /// outputwidth2 // inputwidth2 
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &res2_width_b);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_2_b);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sumadd_no);

//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue1, 
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
 nanosecs [3] = time_end- time_start;
  printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[3]/1000000.0);



////// res2_1 starts \n"////////
printf(" res2_0_1 starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_1);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &outputSignalWidth_pool_1);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_1_1_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_1);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_res2_1); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2_1);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2_1); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_2_1);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_res2_1_1);  /// will be 64 for fully connected layers otherwise same as maskwidth

//errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &str);
printf(" trueeeee");

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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &outputBuffer_pool_1);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_1);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_height_1); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_1);  /// maskwidth_2_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_1_1_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_1);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_1);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_2_1);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

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
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");



kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &biasBuffer_res2_0_1 );
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_1);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_1_1); /// includes padding also   /// inputwidth //
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &res2_1_1_buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_1);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_1);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_1); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2_1);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride2_1);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_2_1);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &res2_height_1_1);      /// outputdim 
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &biasBuffer_res2_0_1);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_2_1);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu1);/// if there is no relu  
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &res2_width_1_2);    /// inputwidth2
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &res2_width_1);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_2_1);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sumadd_no);

//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue1, 
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
 nanosecs [4] = time_end- time_start;
  printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[4]/1000000.0);


////// res2_0_c starts ///////
printf(" res2_0_c starts \n");
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
//	checkErr(errNum, "clCreateCommandQueue");

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &res2_0_b_buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_c);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_cranch2b_cn_c_buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_b_2);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_c);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[0]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_c_1);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_c);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_c);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_c);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_res2_c); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &avg_pool);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2_c);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2_c); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_2_c_1);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_res2_1_c);  /// will be 64 for fully connected layers otherwise same as maskwidth

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
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &res2_0_b_buffer);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_c);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_cranch2b_cn_c_buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_height_c); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_c);  /// maskwidth_2_c acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[0]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_c);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_c);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_c);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_c);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_2_c);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &fc_param_mask); //// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &fc_param_mask_2);  /// fixed parameters for all the layers //
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &fc_param_mask_3);   // fixed parameters for all the layers //
//errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad);
//errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride1);
//printf(" trueeeee");

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
		&event1);
//checkErr(errNum, "clEnqueueNDRangeKernel");


kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &biasBuffer_res2_0_c );
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res2_0_c);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res2_width_c_1); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_2_c);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[0]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_2_c);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_2_c);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_2_c);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_c); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2_c);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride2_c);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_2_c);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &res2_height_c_1);
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &res2_1_1_buffer);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param_2_c);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &res2_width_c_2);
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &res2_width_c);
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_2_c);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sumadd_no1);

//printf(" trueeeee");

   	// Pick the first device and create command queue.
    errNum = clEnqueueNDRangeKernel(
		queue1, 
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
///// many convolution layer starts \\\
 
 clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
   clGetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
 nanosecs [5] = time_end- time_start;
  printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[5]/1000000.0);
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
	checkErr(errNum, "clCreateCommandQueue");
queue3 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

for(int j =0; j<53;j++)
{
//printf(" layer %d %d  starts \n",j,buffer_param[j]);
int f = j - buffer_param[j];
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
     

  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &resbuffer[f]);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res[j]);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &biasBuffer_res[j]);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res_width_inp[j]);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_res[j]);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[j+1]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_res_1[j]);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_res[j]);  /// along the zx direction ///
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_res[j]);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_res[j]);
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &stride_res[j]); // to determine if stride is there or not
errNum |= clSetKernelArg(kernel1 ,10 , sizeof(cl_uint), &pool_conv); //////already defined  ///////
errNum |= clSetKernelArg(kernel1 ,11 , sizeof(cl_uint), &res_avg_pool[j]);
errNum |= clSetKernelArg(kernel1 ,12 , sizeof(cl_uint), &window_check2_res[j]);
errNum |= clSetKernelArg(kernel1 ,13 , sizeof(cl_uint), &pool1_2_res[j]); 
errNum |= clSetKernelArg(kernel1 ,14 , sizeof(cl_uint), &maskheight_res_1[j]);  /// will be 64 for fully connected layers otherwise same as maskwidth
errNum |= clSetKernelArg(kernel1 ,15 , sizeof(cl_uint), &stride_res_1[j]);  /// will be 64 for fully connected layers otherwise same as maskwidth

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
     
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &resbuffer[f]);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res[j]);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &biasBuffer_res[j]);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res_height[j]); /// includes padding also
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_res[j]);  /// maskwidth_res[j] acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[j+1]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_res_1[j]);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_res[j]);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_res[j]);  /// which is kernelwidth in this case
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_res[j]);
errNum |= clSetKernelArg(kernel1 ,9 , sizeof(cl_uint), &maskheight_res_1[j]);  /// will be 64 for fully connected layers otherwise same as maskwidth
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

int t =j;
if( j >=2)
{
   t = j-2;
}
if(buffer_param[j] ==2)
{
  t = j;
}
if(buffer_param[j] ==1)
{
  t = j;
}

kernel1 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
//	checkErr(errNum, "clCreateKernel");
        
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &biasBuffer_res[j] );
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_res[j]);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &res_width_1[j]); /// includes padding also      // inputwidth1
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskwidth_res[j]);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &resbuffer[j+1]);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width_res_1[j]);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width_res[j]);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &height_res[j]);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window_2_res[j]); 
errNum |= clSetKernelArg(kernel1, 9, sizeof(cl_uint), &pad_2_res[j]);        /////
errNum |= clSetKernelArg(kernel1, 10, sizeof(cl_uint), &stride_res[j]);
errNum |= clSetKernelArg(kernel1, 11, sizeof(cl_uint), &pool_conv);      //// already defined ///
errNum |= clSetKernelArg(kernel1, 12, sizeof(cl_uint), &stride_fact_res[j]);
errNum |= clSetKernelArg(kernel1, 13, sizeof(cl_uint), &res_height_1[j]);           ///// outputdim ///
errNum |= clSetKernelArg(kernel1, 14, sizeof(cl_mem), &resbuffer[t]);
errNum |= clSetKernelArg(kernel1, 15, sizeof(cl_uint), &ll3_conv);      /// already defined determines amount of data being written to ////
errNum |= clSetKernelArg(kernel1, 16, sizeof(cl_uint), &stride_param[j]);
errNum |= clSetKernelArg(kernel1, 17, sizeof(cl_uint), &relu_res[j]);///determines  if there is relu or not
errNum |= clSetKernelArg(kernel1, 18, sizeof(cl_uint), &fc_check1);
errNum |= clSetKernelArg(kernel1, 19, sizeof(cl_uint), &res_width_2[j]);         //// outputwidth2  /// inputwidth2
errNum |= clSetKernelArg(kernel1, 20, sizeof(cl_uint), &res_width[j]);         //// outputwidth3    ///used only for padding 
errNum |= clSetKernelArg(kernel1, 21, sizeof(cl_uint), &stride_write_res[j]);
errNum |= clSetKernelArg(kernel1, 22, sizeof(cl_uint), &sum_param[j]);

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
 nanosecs [6+j] = time_end- time_start;
//printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[6+j]/1000000.0);

}


//clgetEventProfilingInfo(event1,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
 //clgetEventProfilingInfo(event1,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
//nanosecs [6+j] = time_end- time_start;
 //printf(" execution time for pool1 is   %0.3f milisecs\n", nanosecs[6+j]/1000000.0);
//printf(" vlaue of data is %d %d %d %d %d %d %d %d  \n", res_avg_pool[53], pool1_2_res[53],pad_2_res[53], input_height_res[53] , res_width_2[51],stride_res[51], stride_res_1[51],sum_param[51]);
//printf(" vlaue of data is %d %d %d %d %d %d %d %d ", stride_write_res[53],res_width_1[53], res_height_inp[53], res_height[53] , res_width_2[53], window_width_res[53],window_2_res[53],res_width_4[53]);

errNum = clEnqueueReadBuffer(
		queue3, 
		resbuffer[53],    ////res2_0_b_buffer      resbuffer[0]
		CL_TRUE,
        0, 
		sizeof(float) * input_height_res[52] * res_height[52]* res_width_4[52],  ///conv1_height * conv1_width *input_height_1  ,  height_pool_1* outputSignalHeight_pool_1 * outputSignalWidth_pool_1, res2_width_a * res2_height_a *input_height_2_a  input_height_res[11] * res_height[11]* res_width_4[11]
		res_fc[0],          /////////  res_reoder_1[i]  res3_0_b[i]3
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

printf("total execution time is \n");
double nanosecs1 = 0 ;
for(int df = 0; df< conv_layers + 6;df++)
{
   nanosecs1 += nanosecs[df]; 
}
printf(" total execution tim is is   %0.3f milisecs\n", nanosecs1/1000000.0);

nanosecs2 += nanosecs1;

time_end1 = clock();

double cpu_time_used= ((double) (time_end1-time_start1))/ CLOCKS_PER_SEC;
 
 //clFinish(queue3);
  //clWaitForEvents(1,&event3);

    //printf(" cpu execution time is %f \n", cpu_time_used);
printf(" true \n");

std::cout<<" output result is   "<<std::endl;
//printf(" conv1_width is 
int r=0;
float max =0 ;
int index  = 0;
for(int i =0 ; i< 1000;i++)
{
  if(res_fc[0][i][0][0][0] > max) 
    {
        max = res_fc[0][i][0][0][0];
 //       printf(" value of index is %d %f \n" , i  ,res_fc[0][i][0][0][0]);
       index = i;
     }
}
int tmp =0;
if(index == label1[df])
{
      tmp = 1;
}
else
{
    tmp  = 0;
}
accuracy += tmp;
Mat image4(cv::Size(224* 3,224* 3), CV_8UC3, cv::Scalar(0));
str << " Image  is " << label[index];
str1 << " time is " << nanosecs1/1000000.0;
str2 << " Frame per secs  is " << (1/(nanosecs2/(1000000.0*(xy * num_image + df+1)))) * 1000;
str3 << "  Accuracy percentage is  " << (accuracy/(xy * num_image + df+1)) * 100 ;
str4 << "Batch Size is " << 1  ;
int j = i * 100 + 5;
cv::putText(image4, str.str(),  cv::Point(224, 15  ), FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,1,0);
cv::putText(image4, str1.str(),  cv::Point(224,45 ), FONT_HERSHEY_PLAIN,1, Scalar(0,255,0),1,8,0);
cv::putText(image4, str2.str(),  cv::Point(224, 65 ), FONT_HERSHEY_PLAIN,1, Scalar(200,200,180),1,8,0);
cv::putText(image4, str3.str(),  cv::Point(224,85 ), FONT_HERSHEY_PLAIN,1, Scalar(255,143,143),1,8,0);
cv::putText(image4, str4.str(),  cv::Point(224, 105 ), FONT_HERSHEY_PLAIN,1, Scalar(255,143,143),1,8,0);
Mat roi = image4(Rect(0,0,224,224));
image1[df].copyTo(roi);
imshow(" resnet_model", image4);
waitKey(1);
if(df == num_image-1 && xy == 3)
{
key = waitKey(0);
}
 printf(" image is  %s %d %f \n", label[index], index, max);
}
}

//printf(" value fo fc8 is %f\n", fc_8[0][703]);
// Output the result buffer//
//for(int w=0;w<1;w++)
{
//for (int z=0; z<16 ;z++)
{
//printf(" value of z is %d\n", z);
//for (int y = 0; y < 4 ; y++)
	{
  //        printf(" value o f y is %d\n", y);
  	//for (int x = 0; x < res_width_4[50] ;x++)
		{
                 // for(int r = 0; r< 3;r++)
                  {
                    //  std::cout << x+y*(pool_output7)+ z*(pool_output7)*(pool_output7) +w << "        ";  
 //std::cout << res_fc[0][z][0][0][0]<< " ";   //conv2[0][y][x][z]  ///  pool1[0 + y * outputSignalWidth_pool_1  * 16 + x * 16 + pad_pool_1+ z]15     res2_1_b[0][y][x][0][r][z]  res_reoder_2[0][y][x][1][r][z]
                  } 
		}
 //std::cout << std::endl;
	}
//	std::cout << std::endl;
}
std::cout<< "one ";
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






 


