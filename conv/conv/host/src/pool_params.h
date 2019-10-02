////////////////////////////
//////pooling  parameters ////
//const unsigned int pool_conv = 0; /// to see uf there is pooling or not //
//// fixed parameters ////////
const int fc_check_pool = 1; //// need to one, to be used in pooling /// /// used of memrd  will be same for all the pooling
const int fc_check_pool1 = 0; //// need to 0 , to be used in pooling /// ///used in memwrite will be same for all  the pooling ///

const unsigned int pool = 1; /// to see uf there is pooling or not //
const int ll3_pool = 1;
const int pool1_pool  = 1; ///always ////
const int pool_width_1  = 3;
//const unsigned int maskWidth_2 = pool_width-1; 
//pooling stride factors /////
const int pad_pool_1 = 2;
const int stride1_pool_1 = 2;  /// actual stride value 
const int stride_pool_1 = 2 ; // used for mem read kernel only/// 
const int stride_fact_pool_1 = 8; /// in memrd kernel to determine if there is stride or nor //
const int stride_impact_pool_1 = 16;
const int stride_pool1 = 2;
const int stride_param_pool_1 = 3;/// needed when you have to write at at last line and output dimesnion is not the multiple fo 2 
//////////////////////
const int maskWidth_pool_1 = 3;
const int kernel_width_pool_1 = 1;
const int window_width_pool_1 =  27;  /// along with x axis ///
const int height_pool_1 = 96;
const int height_pool_1_1 = height_pool_1/ll;
const int window_pool_1 = 27;    // along with y axis ///
const int window_check_pool = height_pool_1;
///////ouput pooling parameters ///
const int outputSignalWidth_pool_1 = 31 ;
const int outputSignalHeight_pool_1 =31 ;
const float outputSignalWidth_pool_1_2 = 31; // just to make it float for argument 3 of memwrite //
float pool1[2][height_pool_1_1][outputSignalHeight_pool_1][outputSignalWidth_pool_1][16] = {0};   // height_pool_1_1*outputSignalHeight_pool_1 * outputSignalWidth_pool_1 * 16]
float pool1_buf[height_pool_1_1][outputSignalHeight_pool_1][outputSignalWidth_pool_1][16] = {0}; 


//// layer 2 pooling ///////

//pooling stride factors /////
const int pad_pool_2 = 1;
const int stride1_pool_2 = 2;  /// actual stride value 
const int stride_pool_2 = 2 ; // used for mem read kernel only/// 
const int stride_fact_pool_2 = 8; /// in memrd kernel to determine if there is stride or nor //
const int stride_impact_pool_2 = 16;
const int stride_pool2 = 2;
//const int pool1_pool  = 1; ///always ////
const int stride_param_pool_2 = 8;/// needed when you have to write at at last line and output dimesnion is not the multiple fo 2 
//////////////////////
const int pool_width_2  = 3;
const int maskWidth_pool_2 = 3;
const int kernel_width_pool_2 = 1;
const int window_width_pool_2 = 13;  /// along with x axis ///
const int height_pool_2 = 256;
const int height_pool_2_1 = height_pool_2/ll;
const int window_pool_2 = 13;    // along with y axis ///
const int window_check_pool_2 = height_pool_2;


///////ouput pooling parameters ///
const int outputSignalWidth_pool_2 = 15 ;
const int outputSignalHeight_pool_2 =15 ;
const float outputSignalWidth_pool_2_2 = 15;
float pool2[2][height_pool_2_1][outputSignalHeight_pool_2][outputSignalWidth_pool_2][16] = {0};
float pool2_buf[height_pool_2_1][outputSignalHeight_pool_2][outputSignalWidth_pool_2][16] = {0};






//// pooling after convolution 5 /////




//pooling stride factors /////
const int pad_pool_3 = 0;
const int stride1_pool_3 = 2;  /// actual stride value 
const int stride_pool_3 = 2 ; // used for mem read kernel only/// 
const int stride_pool3 = 2;
const int stride_fact_pool_3 = 8; /// in memrd kernel to determine if there is stride or nor //
const int stride_impact_pool_3 = 16;  // parameter not needed 
//const int pool1_pool  = 1; ///always ////
const int stride_param_pool_3 = 6;/// needed when you have to write at at last line and output dimesnion is not the multiple fo 2 
//////////////////////
const int pool_width_3  = 3;
const int maskWidth_pool_3 = 3;
const int kernel_width_pool_3 = 1;/// kinda always kernel_width_pool parameter is 1 ////
const int window_width_pool_3 = 6;  /// along with x axis ///
const int height_pool_3 = 256;
const int height_pool_3_1 = height_pool_3/ll;
const int window_pool_3 = 6;    // along with y axis ///
const int window_check_pool_3 = height_pool_3;


///////ouput pooling parameters ///
const int outputSignalWidth_pool_3 = 6 ;
const int outputSignalHeight_pool_3 =6 ;
const float outputSignalWidth_pool_3_2 = 6;
float pool3[2][height_pool_3_1][outputSignalHeight_pool_3][outputSignalWidth_pool_3][16] = {0};


