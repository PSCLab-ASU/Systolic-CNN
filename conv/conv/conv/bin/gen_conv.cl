#define pool_width 16
#define pool_width2 12
#define pool_width3 12
#define pool_width4 18
#define pool_output 10
#define pool_output4 18
#define sr_size 112 // 18*3
#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define ll 1
#define ll1 16  /// 4 * ll ///
#define ll2  8/// 2 * ll b wher ll is defined based on 4 compute units not 8
#define ll4 3 /// 2 * ll1 ////  currently determines level of parallelism
#define ll5 4 /// 2 * ll /// wher ll is supposed to be one ///
#define ll_reuse 3   /// parameter defined for resuablity  //////
#define pe_num 16
typedef struct  mask_struct {
           float16 dat;

}mask_struct;

typedef struct  mask_struct1 {
           float16 dat[16];

}mask_struct1;


typedef struct  inp_struct {
           float dat[ll_reuse * 16];

}inp_struct;

typedef struct  sum_struct1 {
           float dat[32];

}sum_struct1;
typedef struct  sum_struct {
           float dat[ll_reuse];

}sum_struct;

typedef struct  int_struct {
            int dat[4];

}int_struct;

typedef struct  inp_struct1 {
           float16 dat;

}inp_struct1;



channel inp_struct c0 __attribute((depth(700)));
channel mask_struct c1;
channel int_struct c2;
channel int_struct c3;
channel int_struct c4;
channel int c6;
channel int c7;
channel sum_struct c5 ;
channel sum_struct c8[16]  __attribute((depth(400))) ;
channel sum_struct c9 ;
channel sum_struct c10;
channel sum_struct c11 ;

channel sum_struct c16 __attribute((depth(700)));
channel sum_struct c17 __attribute((depth(700)));
channel sum_struct c18 __attribute((depth(700)));
channel sum_struct c19 __attribute((depth(700)));

channel sum_struct c23[4], c24;
channel int_struct c14, c13;
channel int c15;
//channel inp_struct1 c15;
channel int c20, c21, c22;
channel int_struct c12;
//channel int_struct c13;
channel inp_struct c25;
channel mask_struct c26;
channel inp_struct c27;
channel mask_struct c28;
channel inp_struct c29;
channel mask_struct c30;
channel inp_struct c37;
channel int_struct c38;
channel int_struct c39[16];
channel inp_struct c40;
channel mask_struct c41;
channel inp_struct c42;
channel mask_struct c43;
channel inp_struct c44;
channel mask_struct c45;
channel inp_struct1 c46;
channel inp_struct c47;
channel inp_struct c48;
channel inp_struct c49;
channel inp_struct c50;
channel inp_struct c51;
channel mask_struct c52;
channel mask_struct c53;
channel mask_struct c54;
channel mask_struct c55;
channel inp_struct c56;
channel inp_struct c57;
channel inp_struct c58;
channel inp_struct c59;
channel inp_struct c60;
channel inp_struct c61;
channel inp_struct c62;
channel inp_struct c63;
channel mask_struct c64;
channel mask_struct c65  __attribute__((depth(31)));
channel mask_struct c66   __attribute__((depth(63)));
channel mask_struct c67    __attribute__((depth(95)));
channel mask_struct c68  __attribute__((depth(127))) ;
channel mask_struct c69   __attribute__((depth(159))) ;
channel mask_struct c70   __attribute__((depth(191))) ;
channel mask_struct c71  __attribute__((depth(223))) ;
channel mask_struct c72[16];
channel mask_struct c75  __attribute__((depth(225)))  ;
channel mask_struct c76   __attribute__((depth(287))) ;
channel mask_struct c77   __attribute__((depth(319)));
channel mask_struct c78   __attribute__((depth(351))) ;
channel mask_struct c79   __attribute__((depth(383)));
channel mask_struct c80   __attribute__((depth(415))) ;
channel mask_struct c81   __attribute__((depth(447))) ;
channel mask_struct c82   __attribute__((depth(479))) ;
channel inp_struct c83;
channel inp_struct c84;
channel inp_struct c85;
channel inp_struct c86;
channel mask_struct1 c87;
channel mask_struct1 c88;
channel inp_struct c89[17] ;
channel mask_struct c90[16]  __attribute__((depth(479))) ;

constant float coef0[46] = {9.98312401e-01,8.92383765e-01,8.69534866e-01,8.48001507e-01,8.27672857e-01,8.08269896e-01,7.72814246e-01,7.40785193e-01,7.11686616e-01,6.74743320e-01,6.38046300e-01,5.98139529e-01,5.63585746e-01,5.32842946e-01,4.82570938e-01,4.42066574e-01,4.08721176e-01,3.80120836e-01,3.35733988e-01,3.01782553e-01,2.74896454e-01,2.52503409e-01,2.19044754e-01,1.94367577e-01,1.75328514e-01,1.59766323e-01,1.37073713e-01,1.20695464e-01,1.08253750e-01,9.81965345e-02,8.37272488e-02,7.34111523e-02,6.56398695e-02,5.93964327e-02,5.04776032e-02,4.41593533e-02,3.94211944e-02,3.56262849e-02,3.02252062e-02,2.64117530e-02,2.35583854e-02,2.12767794e-02,1.80355644e-02,1.57509127e-02,1.40434261e-02};
// Lrn Kernel
#define LRN_WIN_SIZE        5
#define LRN_MAX_LOCAL_SIZE  (256/VEC_SIZE) // For alexnet the max dim3 size is 256
#define MAN_BITS             23             // Floating point format setting
#define EXP_MASK            0xFF           // Floating point format setting
#define MAN_MASK            0x7FFFFF       // Floating point format setting
#define EXP_STEP_MIN        13             // PWLF table setting
#define EXP_STEP_LOG        0              // PWLF table setting
#define MAN_INDEX_BITS      2              // PWLF table setting
#define MAN_INDEX_MASK      0x03           // PWLF table setting

__kernel void memrd(__global float  * const restrict input, __global float * const restrict mask,const int inputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window, const int stride_conv, const int pool, const int avgpool, const int window_check, const int pool1, const int maskHeight, const int stride_conv1)
{
float16  inp[ll_reuse];
inp_struct inp1;
inp_struct inp2;
int_struct f;
int z =0 ;
int index =0 ;
int v1 =0;
int d =0;
int k = 0;
int pad1 =0;
int_struct iter;
iter.dat[0] = input_height * window * window2;
iter.dat[1] = kernel_width * maskwidth;
iter.dat[2] = maskHeight;
int d_1 = 0;
if(pool ==0)
{
write_channel_intel(c38, iter);
}
int a =0;
int inputWidth1 = inputWidth * 16;
int inputWidth2 = inputWidth1 * inputWidth;
int x =0;
//printf(" value of data is %f %f \n", mask[0], mask[1]);
//printf(" vlaue of inputWidth sis %d %d %d %d %d %d %d %d  \n", input_height,  window,window2, kernel_width, stride_conv, stride_conv1, maskHeight, maskwidth);

while(index !=input_height* window* window2)
{
index++;
//if(pool ==0  )
{
//printf(" value of index is %d %d %d %d %d %d %d %d %d %d, %d %d %d \n", index, window_check, kernel_width, input_height, window, window2,stride_conv, stride_conv1, maskHeight, maskwidth, inputWidth, inputWidth2, inputWidth1);
}
#pragma unroll
for(int j=0;j<ll_reuse;j++)
{
#pragma unroll
for(int i=0;i<16;i++)
{
   inp[j][i] = 0;
}
}
#pragma unroll
for(int i =0;i<ll_reuse * 16;i++)
{
  inp1.dat[i] = 0;
}

f.dat[0] =  z;
f.dat[1] = k ;      ///2 * z * inputWidth1 + k * pool_width; - used to calculate output index
f.dat[2] = d;
f.dat[3] = d_1;
if(pool ==0)
{
 write_channel_intel(c2, f);
 
}
else
{
write_channel_intel(c12, f);
}

if( d>window_check)
{
    a = kernel_width + (d-1) * avgpool;
}
else
{
 a = (d) * pool;
}
int v2 =  stride_conv * (z * inputWidth * pool_width) + k * pool_width * stride_conv1 + a * inputWidth2 * pool1;
//if( z ==0 && k ==0)
{
  //printf(" value fo index is %d \n", index);
}
//if( f.dat[0] == 0 && f.dat[1] == 0 && f.dat[2] ==0  && pool == 0 && f.dat[3] == 0  )
{
//printf(" value of index is %d\n", index);
}
//if(index ==2)
{
//printf(" vlaue of data iis %d %d, %d %d\n ",v2, f.dat[0], f.dat[1], f.dat[2]); 
}
//if( z ==1 && k ==0 && d ==0)
//{
//printf(" value of index is %d\n",index);
//}
//if(index ==2)
//{
//printf(" value of v is %d %d %d %d\n", v2, z, k, d);
//}

for(int r =0 ; r< kernel_width ; r++)
{
int v3 = v2 + r * inputWidth2 ;
int v4 = v3;
x =0 ;
for(int i =0; i<maskHeight;i++) 
{
 v1 =  v3 + ( i) * inputWidth1;
 inp[1]  = (float16)( input[v1+0] , input[v1+1], input[v1+2] , input[v1+3], input[v1+4] , input[v1+5], input[v1+6] , input[v1+7],input[v1+8] , input[v1+9], input[v1+10] , input[v1+11], input[v1+12] , input[v1+13], input[v1+14] , input[v1+15] );
v1 = v1+ 16;
inp[0]  = (float16)( input[v1+0] , input[v1+1], input[v1+2] , input[v1+3], input[v1+4] , input[v1+5], input[v1+6] , input[v1+7],input[v1+8] , input[v1+9], input[v1+10] , input[v1+11], input[v1+12] , input[v1+13], input[v1+14] , input[v1+15] );

for(int y=0;y<(maskwidth);y++)
{
 v1 = v1+16; 
#pragma unroll
for(int  e= 2; e>0;e--)
 {
     inp[e] = inp[e-1];
  }
inp[0]  = (float16)( input[v1+0] , input[v1+1], input[v1+2] , input[v1+3], input[v1+4] , input[v1+5], input[v1+6] , input[v1+7],input[v1+8] , input[v1+9], input[v1+10] , input[v1+11], input[v1+12] , input[v1+13], input[v1+14] , input[v1+15] );



int f =0;
int b =0;

#pragma unroll
for(int e =0;e<ll_reuse;e++)
{
  
  #pragma unroll
  for(int j =0 ;j<16;j++)
    {    
      inp1.dat[ b*16 + j] = inp[e][j];
           }
     b++;
  }
//if( r ==0 && i ==0 && y ==1 && index ==1)
{
 // printf(" value of input is %d %f\n", v1, inp1.dat[0]);
}
//if( y == 0 && i ==0 && r ==0 && index ==1)
{
// printf(" value of inp is %f %f %f %f %d %d %d \n", inp1.dat[0], inp1.dat[1], inp1.dat[16], inp1.dat[17], maskHeight, maskwidth, kernel_width);
}
if(pool ==0)
{
write_channel_intel(c89[0], inp1);
}
else
{
write_channel_intel(c47, inp1);
}
}
}
}


if( k== window2-1) // window2- -1
{
z++;
k = 0;
}
else
{
k++;
}
if ( z == window) // window
{
   d++;
   z = 0; 
}
}
}




__kernel void mask_read(__global float  * const restrict input, __global float16 * const restrict mask,const int inputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window, const int maskHeight, const int fc_param, const int fc_conv, const int fc_param2)
{
float16  mask_loc;
float16  mask_loc1;
float16  mask_loc2;
float16  mask_loc3;
float16  mask_loc4;
float16  mask_loc5;
float16  mask_loc6;
float16  mask_loc7;
float16  mask_loc8;
float16  mask_loc9;
float16  mask_loc10;
float16  mask_loc11;
float16  mask_loc12;
float16  mask_loc13;
float16  mask_loc14;
float16  mask_loc15;

float16 mask_loc34[pe_num];
int16 mask32;
int16 mask33;
mask_struct mask_loc16;
mask_struct mask_loc17;
mask_struct mask_loc18;
mask_struct mask_loc19;
mask_struct mask_loc20;
mask_struct mask_loc21;
mask_struct mask_loc22;
mask_struct mask_loc23;
mask_struct mask_loc24;
mask_struct mask_loc25;
mask_struct mask_loc26;
mask_struct mask_loc27;
mask_struct mask_loc28;
mask_struct mask_loc29;
mask_struct mask_loc30;
mask_struct mask_loc31;
mask_struct mask_loc35[pe_num];

int index = 0;
//int d =0;
int v = 0;
int flag = 0;
int v2 =0;
int v3 = 0;
int v4 = 0;
write_channel_intel(c6, fc_conv);
int param = kernel_width * maskwidth * maskHeight;
//mask_loc[0] = (1 << 22) + (1 & 0x000000ff) ;
//printf(" value of mask is %f\n", mask_loc[0]); 
//printf(" value of fc_conv is %d %d\n", fc_conv, fc_param);
while(index !=input_height* window * window2)
{
index++;
#pragma unroll
for(int i =0;i<pe_num;i++)
{
mask_loc[i] = 0;
mask_loc1[i] = 0;
mask_loc2[i] = 0;
mask_loc3[i] = 0;
mask_loc4[i] = 0;
mask_loc5[i] = 0;
mask_loc6[i] = 0;
mask_loc7[i] = 0;
mask_loc8[i] = 0;
mask_loc9[i] = 0;
mask_loc10[i] = 0;
mask_loc11[i] = 0;
mask_loc12[i] = 0;
mask_loc13[i] = 0;
mask_loc14[i] = 0;
mask_loc15[i] = 0;
mask_loc34[i] = 0;
}
//printf(" value of index is %d %d %d \n", index, fc_conv, fc_param);
//int d = read_channel_intel(c6);

int_struct f = read_channel_intel(c2);
v = (f.dat[2] * param) * 16 ;
v2 = v + 4 * param;
v3 = v + 8 * param;
v4 = v + 12 * param;

int s =0;
//printf(" value of index is %d\n", index);
for(int r =0;r< kernel_width; r++)
{
      
   
   
  for(int i =0;i< maskwidth; i++)
   {
   //    flag = i & 0x01;
     
     for(int x =0;x < maskHeight;x++)
      { 
       // considering fc_param need ot bt 4
         #pragma unroll 
      for(int y =0; y<4;y++)
      { 
       mask_loc34[y] = mask[  v  + y]; 
       }
         v = v + 4; //// fc_conv is always 1  
        if( fc_conv ==1)
        {
        #pragma unroll
       for(int y =0 ; y<4;y++)
       {
       mask_loc34[y+4] = mask[  v2 + y ]; 
       mask_loc34[y+8] = mask[ v3 + y ];  
       mask_loc34[y+12] = mask[  v4 + y ]; 
       }
       v2 = v2 + 4;
              v3 = v3 + 4; //// fc_conv is always 1  
       
       v4 = v4+ 4;
         #pragma unroll
       for(int y =0; y<4;y++)
         {
        mask_loc35[y].dat =  mask_loc34[y]; 
         }          
      }
         else
        {     
               #pragma unroll
             for(int ef =0;ef<16;ef++)
              {
              
               mask32[0] =  ((int) mask_loc34[0][ef]) ;
               mask32[1] =  ((int) mask_loc34[1][ef]) ;
               mask32[2] =  ((int) mask_loc34[2][ef]) ;
               mask32[3] =  ((int) mask_loc34[3][ef]) ;
          
              mask_loc35[1].dat[ef] =  (((mask32[0] & 0x00000800) ==  2048)? ~(mask32[0] & 0x000007f0) : mask32[0] & 0x000007f0) >> 6;
               mask_loc34[5][ef] = (((mask32[1] & 0x00000800) == 2048)? ~(mask32[1] & 0x000007f0) : mask32[1] & 0x000007f0) >> 6;
               mask_loc34[9][ef] = (((mask32[2] & 0x00000800) == 2048)? ~(mask32[2] & 0x000007f0) : mask32[2] & 0x000007f0) >> 6;
               mask_loc34[13][ef] = (((mask32[3] & 0x00000800) == 2048)? ~(mask32[3] & 0x000007f0) : mask32[3] & 0x000007f0) >> 6;
           
            mask_loc35[2].dat[ef] = (((mask32[0] & 0x00020000) == 131072 )? ~(mask32[0] & 0x0001f000) : mask32[0] & 0x0001f000) >> 12;
             mask_loc34[6][ef] = (((mask32[1] & 0x00020000) == 131072 )? ~(mask32[1] & 0x0001f000) : mask32[1] & 0x0001f000) >> 12;
             mask_loc34[10][ef] = (((mask32[2] & 0x00020000) == 131072 )? ~(mask32[2] & 0x0001f000) : mask32[2] & 0x0001f000) >> 12;
             mask_loc34[14][ef] = (((mask32[3] & 0x00020000) == 131072 )? ~(mask32[3] &0x0001f000) : mask32[3] & 0x0001f000) >> 12;
               
            
              
            mask_loc35[3].dat[ef] =( (mask32[0] & 0x00800000)== 8388608? ~(mask32[0] & 0x007f0000):(mask32[0] & 0x007f0000)) >> 18;
            mask_loc34[7][ef] =  ( (mask32[1] & 0x00800000)==  8388608? ~(mask32[1] & 0x007f0000):(mask32[1] &0x007f0000)) >> 18; 
            mask_loc34[11][ef] = ( (mask32[2] & 0x00800000)== 8388608? ~(mask32[2] & 0x007f0000):(mask32[2] & 0x007f0000)) >> 18;
            mask_loc34[15][ef] = ( (mask32[3] & 0x00800000)== 8388608? ~(mask32[3] & 0x007f0000):(mask32[3] & 0x007f0000)) >>18;     
               //      if(index ==1 && x ==0 && i ==0 && r ==0)
               //    ((mask32[0] & 0x0000001f) > 0) ? printf(" true how"): printf( " false");
 
               mask_loc35[0].dat[ef] = ((mask32[0] & 0x00000020) == 32)? ~(mask32[0] & 0x0000001f) : mask32[0] & 0x0000001f;  
               mask_loc34[4][ef] = ((mask32[1] & 0x00000020) == 32)? ~(mask32[1] & 0x0000001f) : mask32[1] & 0x0000001f;
               mask_loc34[8][ef] = ((mask32[2] & 0x00000020) == 32)? ~(mask32[2] & 0x0000001f) : mask32[2] & 0x0000001f;
               mask_loc34[12][ef] = ((mask32[3] & 0x00000020) == 32)? ~(mask32[3] & 0x0000001f) : mask32[3] & 0x0000001f; 
               
 
             }
    // if(index == 1  )
             {
         //     printf(" value of data is %d %f %f %f %f %f  \n", i * 16,  mask_loc[0], mask_loc16.dat[0],mask_loc17.dat[0],mask_loc18.dat[0],mask_loc19.dat[0] );
             }
            }
               
       #pragma unroll  
       for(int y =4 ;y<pe_num;y++)
        {
         #pragma unroll
       for(int ef=0;ef<16;ef++)
       {
            
       
       
       mask_loc35[y].dat[ef] =  mask_loc34[y][ef]; 
              }
       }
      #pragma unroll
      for(int i = 0; i<pe_num;i++)
         {
       write_channel_intel(c90[i], mask_loc35[i]);  
        }
      //  if( i ==0 && r ==0 && index ==1 && x ==0)
         {
       //    printf(" value of data is %f %f %f %d %d %d", mask_loc16.dat[0], mask_loc16.dat[1], mask_loc16.dat[2], maskHeight, maskwidth, kernel_width);
           }
            
       }
         
       }    
   }
        
 write_channel_intel(c14, f);
}

}





__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void channel_ker()
{
     int index =0;
     
   int fc = read_channel_intel(c6);
     int_struct iter1;
     iter1 = read_channel_intel(c38);
      #pragma unroll
      for(int i =0 ;i<pe_num;i++)
        {
            write_channel_intel(c39[i], iter1);
        }
 //printf(" value of fc is %d\n", fc);
     int iter = iter1.dat[0];
     int kernel_width = iter1.dat[1];
     int maskwidth = iter1.dat[2];
      inp_struct inp1;
      mask_struct mask_loc1[2];
      int fc_check;
      inp_struct inp2;
      mask_struct mask_loc2[2];
     int_struct f;      
    while( index != iter)
        {
        index++;
        
 
int s =0;
      
       for(int i =0; i< kernel_width;i++)
     {
      for(int j =0; j < maskwidth ;j++)
       {
 
          int x = 0; 
          int w =0;
      #pragma unroll   
     for(int r =0;r<ll5; r++)
         {
          w = r *(fc+1);
         #pragma unroll
          for(int y =0; y<2; y++)
          { 
              
        //     mask_loc2[0].dat[r*2 + y] = mask_loc1[0].dat[  w];
             x = w + 2 * fc;
          //   mask_loc2[1].dat[r*2 + y] = mask_loc1[0].dat[  x+ll5  ];
              w = w+ fc;
          }
         }
          
       } 
      }     
f = read_channel_intel(c14);
write_channel_intel(c3, f);
 }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))	
__attribute__((num_compute_units(pe_num)))	
__kernel void conv()
{
int_struct iter;
int compute_id = get_compute_id(0);
iter = read_channel_intel(c39[compute_id]);
inp_struct  inp1;
inp_struct  inp2;
mask_struct mask_loc;
mask_struct mask_loc1;
float inp_shift[sr_size];
float  sum[2][16];
sum_struct  sum1;
sum_struct sum_acc;
int index = 0; 
int flag = 0;
float sum2[ll_reuse];
float  sumx[8];
float  sumx1[ll_reuse] ;
float sumx2[ll_reuse];
float sumx3[ll_reuse] ;
float sumx4[ll_reuse] ;
float sumx5[ll_reuse] ;
float sumx6[ll_reuse] ;
float sumx7[ll_reuse] ;
float sumx8[ll_reuse] ;
float sumx9[ll_reuse] ;

while(index != iter.dat[0])
{
index++;

#pragma unroll
for(int e=0;e<ll;e++)
{
#pragma unroll
for(int j =0;j<2;j++)
{
#pragma unroll
for(int i=0; i< 16; i++)
{
   sum[j][i] = 0.0f;
}
}
}       
#pragma unroll
      for(int r =0; r<ll_reuse;r++)
     {
     sum_acc.dat[r] = 0.0f;
     sumx1[r] = 0;
     sumx2[r] = 0;
     sumx3[r] = 0;
     sumx4[r] = 0;
     sumx5[r] = 0;
     sumx6[r] = 0;
     sumx7[r] = 0;
     sumx8[r] = 0;
     sumx9[r] = 0;  
     sum2[r] =0 ;
      }  
   
int r =0;

for(int x =0;x< iter.dat[1];x++)
{
   for(int j =0; j< iter.dat[2]; j++)
        {
          inp1 = read_channel_intel(c89[compute_id]);
          mask_loc = read_channel_intel(c90[compute_id]);
          int e =0;
            
          #pragma unroll 
         for(int d =0; d< ll_reuse; d++)
         {
          sumx1[d] = 0;
          sumx2[d] = 0;
          sumx3[d] = 0;
          sumx4[d] = 0;
          }  
          
          #pragma unroll
       for(int d =0 ; d< ll_reuse;d++)
          {
         #pragma unroll
         for(int i = 0; i< 4;i++)
          {
            sumx1[2-d] +=  inp1.dat[i + 16 * d] *  mask_loc.dat[i] ;
            sumx2[2-d] +=  inp1.dat[i+4 + 16 * d] * mask_loc.dat[i + 4] ;
            sumx3[2-d] +=  inp1.dat[i+8 + 16 * d] * mask_loc.dat[i+8] ;
            sumx4[2-d] +=  inp1.dat[i+12 + 16 * d] *mask_loc.dat[i+12];
            
         }
        }
             for(int d =0 ;d < ll_reuse;d++)
              {  
              sumx5[d] = sumx1[d] + sumx2[d]; 
              sumx6[d] = sumx3[d] + sumx4[d];
              sumx7[d] = sumx5[d] + sumx6[d];
              sumx8[d] = sumx7[d] + sum2[d];
              sum2[d]  = sumx8[d];
               }
       //if( index == 112)
        {
        //  printf(" value of data is at %d %d %d  \n", x/maskwidth * 16, x%maskwidth, j); 
       //printf(" value of data is%d %f %f  %f, %f %f\n", x%maskwidth, inp1.dat[32], mask_loc.dat[0], mask_loc.dat[1], sum2[0], sum2[1]); 
         // for(int f =0; f<16;f++)
            {
           //   printf(" %f  %f  \n", inp1.dat[32+f], mask_loc.dat[f]);
            }
        }
              inp2 = inp1;
              mask_loc1 = mask_loc;
              if(compute_id < 15)
              write_channel_intel(c89[compute_id+1], inp2);
              
  }
          
             
  
}
#pragma unroll
for(int i =0 ;i<3;i++)
{
sum_acc.dat[i] = sum2[i];
}
write_channel_intel( c8[compute_id], sum_acc);
}
}








__kernel void memwrite(__global float16  * const restrict bias4, __global float * const restrict mask,const int outputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window, const int pad, const int stride, const int pool, const int stride_fact, const int outputdim, __global float * const restrict bias, const int ll3_conv, const int stride_param, const int relu, const int fc_check, const int  outputWidth2, const int outputWidth3, const int stride_write, const int sumadd) 
{
// pad1 only used before lrn
int index =0;
int inputWidth1 = outputWidth  ; //// here inputwidth coming from host side is basically output 
int inputWidth2 = outputWidth2 ;
sum_struct sum1[pe_num];
float sum2[pe_num];
float sum3[pe_num];
int e =0;
int s =0;
float sum4;
float sum5;
int ez = 0;
int a =0;
int y =0;
s = 2;
uint4_t ea =0;
int e1 =0;
int_struct f ;
float16 bias1 ;// =  { 2.149, 1.833, -1.11, 1.75, 1.19, -0.712, 1.27, 2.45, 5.0852, -0.9441, 1.74, 4.4, -1.030, 0.082, 3.3098, 1.24 };
float16 bias2;
uint2_t ef = 0;
int sum6;
//printf(" value of ll3_conv is %d\n", ll3_conv);
//printf(" value of data is %d %d %d %d %d \n", inputWidth2 , outputdim, inputWidth1, input_height, pad1);
//printf(" value of input_height is %d %d \n", stride_param, ll3_conv);
int stride_block;
while(index !=input_height* window * window2 * ll_reuse )
{
index++;


int c = f.dat[2];
if(ez == ll_reuse)
{
ez = 0;
}
if( ez ==0)
{
f = read_channel_intel(c3);
 e =  (f.dat[2]) * pe_num * inputWidth2 * outputdim +  f.dat[0] * pe_num * inputWidth1 + f.dat[1] * pe_num * stride_write  + pe_num * (outputWidth* pad) + pe_num *  pad ;

a = f.dat[2];
bias2 = bias4[a];
y = 0;
ea = 0;
stride_block = stride_write -  1;
//printf(" value of e is %d %d %f %d\n", e, stride_fact, inputWidth1, outputdim);
//if( kernel_width ==1)
{
//printf(" value of index is %d\n", index);
}
#pragma unroll
for(int g =0; g<pe_num;g++)
{
 {
 sum1[g] = read_channel_intel(c8[g]);
  }
 }
}
//if( e ==16)
{
  //printf(" true \n");
}
//printf(" value of index is %d %d %d %d %d %d %d %d %d %d \n", index,kernel_width, input_height, window, window2,pad, inputWidth2, outputdim, stride_param,ea);
// printf(" value of index, input_height, window, window2, ll4 is %d %d %d %d %d %d \n", index,ez,  input_height, window, window2, ll4);
ez++;
//if( f.dat[1] == 1 && f.dat[0] ==0 && f.dat[2] ==0)
{
 //printf(" value of index is %d %d %d %d \n", e,index,  window, window2);
}
if(sumadd == 0)
{
#pragma unroll
for(int i =0 ;i<pe_num;i++)
{
bias1[i] = bias[e+i] ;
//printf(" value of index and e is %d %d \n", index, e);
}
}
else
{
bias1 = 0;
}
//printf(" value of ez is %d %d\n", ez, ea);
int z =0;
{
#pragma unroll
for(int i =0;i<pe_num; i++)
{
 sum2[i] = 0; 
}
//printf(" value of ea is %d\n", ea);


//#pragma unroll
//for(int r =0; r<ll;r++)
{
int j =0;
//int y =0;
//int a =0;
int d =0;
int t =0;
//#pragma unroll
//for(int j =0;j<3;j++)
{
int rd =0;
#pragma unroll
for(int i =0;i<pe_num;i++)
{
  sum5 =  sum1[i].dat[ea] + bias1[i] + bias2[i] ;
  sum2[i ] = ( ea >= stride_param && f.dat[1] == window2-1) ? 0: sum5;  /// done to avoid out of ordering as was in previous case //    sum2[i] = ((f.dat[0] == window-1 && t > stride_param+1 && fc_check ==0)) ? 0 : sum5 // this expression could be used not sure though;
t++;
  
 }   //// only will work for uimages of odd dimesntion can one control signal to say if it is odd or even	
   /// done if fc_check will be one for stride convoluton then need that 
}
}
int j =0;

#pragma unroll
for(int i =0; i<pe_num;i++)
{
  sum4 = sum2[i] ;
  sum5 = (sum4>0 )? sum4: sum4 * relu; 
  sum3[i] = sum5;
j++;
}

//if(index ==2)
//{
//printf(" value of sum is %f\n", sum3[0]);
//}

//if( f.dat[0] == 0 && f.dat[1] == 0 && f.dat[2] ==0 && f.dat[3] == 0 && window2 == 9 )
{
//printf(" value of index is %d\n", index);
}

//printf(" value of ea is %d\n", ea);
//if(index == 25 & ea == 0 && window2 == 9)
{
//printf(" value of e is %d %d %d %d %d %d \n", e, inputWidth1, f.dat[0], f.dat[1], f.dat[2], f.dat[3]);
 
//for(int i =0 ;i<16;i++)
 {
  //printf(" value of sum is %f \n", sum3[i]);
 }
}
//if(index == 1 )
{
  //printf(" value of index is %d\n", index);
  //for(int i =0;i< 16;i++)
   {
    // printf(" value of sum2 is %d %f\n",i,  sum3[i]);
   }
}
//for(int i =0;i<
/// f.dat[0] = z. f.dat[1] = k,  f.dat[2] = d
//printf( " value of e is %d\n", e);
// printf(" value of index is %d\n", index);

//printf(" value of sum is  %f\n", sum1.dat[0]);
 
//printf(" value of j is " );
int r = 16;
int x = 0;

int b =0;
int ed = 0;
//#pragma unroll
//for(int j =0 ; j<3;j++)
{
 
//for(int y =0;y< pad; y++)
{
#pragma unroll 
for(int i =0;i<pe_num;i++)
{ 
output[e + i ] = sum3[i];    ;//e+  ea * inputWidth2+ ed
//if( e + i == 3649)
{
// printf(" value of results is %d %d %d %d, %d %f \n",ea, f.dat[0], f.dat[1], f.dat[2],  index, sum3[i]);
}
ed ++;
}
}
}
if (stride_block ==0)
{
e = e;
}
else
{
 e= e + pe_num;
 stride_block = stride_block -1;
}
ea++;
}
}
}









// max pooling ////
__kernel void pool(__global float  * const restrict input, __global float * const restrict mask,const int outputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int pool_height, const int window, const int pad)
{
int index =0; 
inp_struct  inp;
int_struct f;
int e =0;
float pool_res[32];
sum_struct1 pool_acc;
const int outputWidth1 = outputWidth * outputWidth;
//printf(" value of pad is %d \n", pad);
while( index != pool_height * window * window2) 
{
index++;
#pragma unroll
for(int i =0;i<32;i++)
 {
     pool_res[i] = 0;
  }
#pragma unroll
for(int i =0;i<32;i++)
{
   pool_acc.dat[i] = 0;
}

f = read_channel_intel(c12);

 
for(int i =0; i< maskwidth;i++)
{
 
 
   for(int j =0;j< maskwidth;j++)
   { 
       inp = read_channel_intel(c47);
        
        pool_res[0] = max(inp.dat[(ll_reuse-1) * 16 + 0], pool_acc.dat[0]);
        pool_res[1] = max(inp.dat[(ll_reuse-1) * 16 + 1], pool_acc.dat[1]);
        pool_res[2] = max(inp.dat[(ll_reuse-1) * 16 + 2 ] , pool_acc.dat[2]);
        pool_res[3] = max(inp.dat[(ll_reuse-1) * 16 + 3 ], pool_acc.dat[3]);
        pool_res[4] = max(inp.dat[(ll_reuse-1) * 16 + 4 ], pool_acc.dat[4]);
        pool_res[5] = max(inp.dat[(ll_reuse-1) * 16 + 5 ], pool_acc.dat[5]);
        pool_res[6] = max(inp.dat[(ll_reuse-1) * 16 + 6], pool_acc.dat[6]);
        pool_res[7] = max(inp.dat[(ll_reuse-1) * 16 + 7], pool_acc.dat[7]);
        
        pool_res[8] = max(inp.dat[(ll_reuse-1) * 16 + 8], pool_acc.dat[8]);
        pool_res[9] = max(inp.dat[(ll_reuse-1) * 16 + 9], pool_acc.dat[9]);
        pool_res[10] = max(inp.dat[(ll_reuse-1) * 16 + 10], pool_acc.dat[10]);
        pool_res[11] = max(inp.dat[(ll_reuse-1) * 16 + 11], pool_acc.dat[11]);
        pool_res[12] = max(inp.dat[(ll_reuse-1) * 16 + 12], pool_acc.dat[12]);
        pool_res[13] = max(inp.dat[(ll_reuse-1) * 16 + 13], pool_acc.dat[13]);
        pool_res[14] = max(inp.dat[(ll_reuse-1) * 16 + 14], pool_acc.dat[14]);
        pool_res[15] = max(inp.dat[(ll_reuse-1) * 16 + 15], pool_acc.dat[15]);
      #pragma unroll
      for(int x=0;x< 16;x++)
     {
         pool_acc.dat[x] = pool_res[x];
     }
       //if(index ==756  && i ==0)
         {
         //  printf(" value of data is %f %f %f \n", inp.dat[38], inp.dat[39], inp.dat[40]);
          }
  }
}
e = f.dat[2] * outputWidth1 * 16 +  f.dat[0] * 16  * outputWidth + f.dat[1] * 16 + 16 * (outputWidth* pad) + 16 *  pad ;
//if( f.dat[2] == 0 && f.dat[3] == 0 && f.dat[0] ==0 && f.dat[1] == 0)
{
 //printf(" value of index is %d\n", index);
}
//if(index == 27)
{
//printf("value of e is %d\n", e);
//for(int i =0 ;i<16;i++)
  {
  //  printf(" value of data is %d %f\n",i,  pool_acc.dat[i]);
  }
}
#pragma unroll 
for(int i =0; i<16;i++)
{
output[ e + i] = pool_acc.dat[i];
//if( e+i == 17 )
 {
  //  printf(" value of index is %d\n", index);
  }
}
}
}




__kernel void lrn_read(__global float  * const restrict input, __global float * const restrict mask,const int outputWidth, const int maskwidth,__global float16 * const restrict output, const int kernel_width, const int window2, const int input_height, const int window, const int pad, const int outputHeight)
{
inp_struct1 inp;
int index = 0;
int inputWidth1 = outputWidth * 16;
int inputWidth2 = inputWidth1 * outputHeight;
//int k =  get_global_id(0);
//int z =  get_global_id(1);
int k =0;
int z =0;
while (index != window * window2)
{
index++;
int s = 0;
int_struct f;
//printf(" value of k is %d %d %d %d %d \n", k, kernel_width, outputWidth, outputHeight, pad);
int e =   k* pool_width + z* inputWidth1  + pad + pad * inputWidth1;
write_channel_intel(c15,e);
int x =0;

int v2=  z * inputWidth1+ k * pool_width;
//if( e ==16)
{
//printf(" value of k is %d %d %d %d %d %d %d \n", k, z, v2, kernel_width, outputWidth, outputHeight, pad);
}
for(int r =0;r<kernel_width;r++)
{   
  int v1 = v2 +  r * inputWidth2;
      inp.dat = (float16) (input[v1 +0], input[v1 +1], input[v1 +2], input[v1 +3], input[v1 +4], input[v1 +5], input[v1 +6], input[v1 +7] , input[v1 +8], input[v1 +9], input[v1 +10], input[v1 +11], input[v1 +12], input[v1 +13], input[v1 +14], input[v1 +15]);
 
       write_channel_intel(c46, inp);
}
if(k == window2-1)
{
  k= 0;
  z++;
}
else
{
  k++;
}	
}
}
__kernel void lrn(__global float  * const restrict input, __global float * const restrict mask,const int outputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window, const int pad, const int outputHeight)
{
///  b =  a * (  1 + sum(a^2)) ^0.75
float16 sum =0; 
float inp1[32];
inp_struct1 inp[3];
int index =0 ;
int x= 0;
int outputdim = outputWidth * outputHeight * 16 ;
while(index != window * window2)
{
index++;
//printf("\n value of outputdim is %d\n", outputdim);
//int z = get_global_id(0);
//printf(" value of z i s%d\n ", z);
float16 sum1;
        int          *convert_ptr;
	int          expo;
	uint         manti;
	uint         addr_1[16], addr_2[16], addr[16];
	float16        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
	short        lrn_cnvt, lrn_cnvt2;
float sum3 =0;
#pragma unroll
for(int i =0;i< 2;i++)
{
#pragma unroll
for(int j =0;j< 32;j++)
{
   inp1[j] = 0;
}
}
x= 0;
int e = read_channel_intel(c15);
#pragma unroll
for(int i =0 ;i<3;i++)
{
#pragma unroll
for(int j =0; j<16;j++)
{ 
  inp[i].dat[j]= 0;
}
}
#pragma unroll
for(int i =0; i<16;i++)
{
  sum[i] = 0;
} 
for(int r =0; r< kernel_width;r++)
{ 
//if( e ==16 && r ==0)
{
  //printf(" value of x is %d\n", x);
     
}

for(int i =0; i< 2-x;i++)
  {
         inp[2].dat  = inp[1].dat; 
         inp[1].dat  = inp[0].dat;

      if( r < kernel_width-1)
       {
           inp[0]  = read_channel_intel(c46);    // square if of each number //
          // if( i ==0 && e ==16 && r ==0)
             {
            //    printf(" true\n");
              }
         }
       else
        {
           #pragma unroll
          for(int ef =0;ef<16;ef++)
            {
         inp[0].dat[ef] = 0;      
            }
       
        }
          
        
       #pragma unroll
    for(int y =0; y<16;y++)
      {
         inp1[y] = inp1[y+16];
      }
      #pragma unroll  
      for(int y = 0;y<16;y++)
      {
        inp1[y+16] = inp[0].dat[y] * inp[0].dat[y] ;
       } 
         
        
       }
         
         sum[0] = inp1[0] + inp1[1] + inp1[2];
         sum[1] = sum[0] + inp1[3];
         sum[2] = sum[1] + inp1[4];
           #pragma unroll
          for(int i =0; i<13;i++)
           {
             
              sum[i+3] = sum[i+2] + inp1[i+5]- inp1[i];
             }
            
             #pragma unroll
           for(int i =0; i<16;i++)
              {
                sum3 = sum[i];
                convert_ptr = (int*) (&sum3);
              
		expo = ((*convert_ptr >> MAN_BITS)) - 127;
		  
                manti = ((*convert_ptr) & MAN_MASK); //does not include the implicit 1

		addr_1[i] = ((expo-EXP_STEP_MIN)>>EXP_STEP_LOG)<<MAN_INDEX_BITS;
		addr_2[i] = (manti>>(MAN_BITS-MAN_INDEX_BITS) & MAN_INDEX_MASK)+1;
		
                

                if(expo<EXP_STEP_MIN)
			addr[i] = 0; // use the first segment
		else
			addr[i] = addr_1[i]+addr_2[i];
              // if(z ==2)
                 {
                //    printf(" value of addr is %d %d %d %f %f \n", addr[1], addr_1[1] , addr_2[1], coef0[addr[1]], inp1[2][2]);
                  }
                }
 #pragma unroll
 for(int i =0; i<16;i++) 
  {
   int addr_3 = addr[i];
  lrn_tmp[i] = coef0[addr_3];
  sum1[i] = lrn_tmp[i] * inp[1].dat[i];
 //  if ( r ==0 && e == 16 )
     {
   //      printf(" value of data is %f %f %f \n", inp[2].dat[i], inp[1].dat[i], inp[0].dat[i]);
      }
  // printf(" addr_3 is %d\n", addr_3);
   }
  
 //if(r ==0 && z ==0)
      {
         //for(int i = 0;i<5;i++)	
          {
   //        for(int j =0; j<16;j++)
            {
     //         printf(" value of sum is %d,  %f \n", j, sum1[j]);
           }
          }

       }

// if( r == 0 && z ==0)
   {
  //   for(int i =0; i< 5;i++)
     {
    //   for(int j =0;j<16;j++)
        {
      //     printf(" value of input is %d  %f \n",i *16 + j,  inp1[i][j]);
     }
   }
   //printf(" value of sum is %f\n", sum[2]);
  
  }
 //printf(" value of e is %d\n", e + r * outputdim);
//  if( e + r * outputdim == 192)
  {
 // printf(" value of e is %d\n", e + r * outputdim);
   }
  x = 1;
//if( e ==16 && r ==0)
{
 //for(int i =0 ;i<16;i++)
    {
   //    printf(" value of sum is %f\n", sum1[i]);
   }
}
//if(index ==1 )
{
  //printf(" value of sum is %f\n", sum1[0]);
}
for(int i =0; i< 16;i++)
output[  e+ r * outputdim +i ] = sum1[i];
}
}
}


