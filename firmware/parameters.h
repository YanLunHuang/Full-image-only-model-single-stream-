#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w15.h"
#include "weights/b15.h"
//#include "weights/w18.h"
#include "weights/b18.h"
//#include "weights/w22.h"
#include "weights/b22.h"
//#include "weights/w25.h"
#include "weights/b25.h"
//#include "weights/w29.h"
#include "weights/b29.h"
//#include "weights/w32.h"
#include "weights/b32.h"
//#include "weights/w36.h"
#include "weights/b36.h"
//#include "weights/w39.h"
#include "weights/b39.h"
#include "weights/w42.h"
#include "weights/b42.h"

//hls-fpga-machine-learning insert layer-config
// up_sampling2d
struct config2 : nnet::resize_config {
    static const unsigned height = 56;
    static const unsigned width = 11;
    static const unsigned data_transfer_out = 4;
    static const unsigned n_chan = 4;
    static const unsigned new_height = 56;
    static const unsigned new_width = 55;
};

// q_activation
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = 12320;
    static const unsigned data_transfer_out = 4;
    static const unsigned n_chan = 4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_table_t table_t;
};

// zp2d_q_conv2d_batchnorm
struct config45 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 55;
    static const unsigned data_transfer_out = 4;
    static const unsigned n_chan = 4;
    static const unsigned out_height = 60;
    static const unsigned out_width = 59;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// q_conv2d_batchnorm
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 5;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config4 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 60;
    static const unsigned in_width = 59;
    static const unsigned data_transfer_in = 4;
    static const unsigned data_transfer_out = 16;
    static const unsigned n_chan = 4;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 55;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 47;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 9;
    static const unsigned min_width = 9;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_height * config4::filt_width> config4::pixels[] = {1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,7340032,15728640,32505856,31457280,29360128,25165824,16777216};

// q_conv2d_batchnorm_linear
struct linear_config5 : nnet::activ_config {
    static const unsigned n_in = 49280;
    static const unsigned data_transfer_out = 16;
    static const unsigned n_chan = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_lineartable_t table_t;
};

// q_activation_1
struct relu_config6 : nnet::activ_config {
    static const unsigned n_in = 49280;
    static const unsigned data_transfer_out = 16;
    static const unsigned n_chan = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_1_table_t table_t;
};

// max_pooling2d
struct config7 : nnet::pooling2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 55;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned data_transfer_out = 16;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef max_pooling2d_default_t accum_t;
};

// zp2d_q_conv2d_batchnorm_1
struct config46 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned data_transfer_out = 16;
    static const unsigned n_chan = 16;
    static const unsigned out_height = 30;
    static const unsigned out_width = 29;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_1
struct config8_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 24;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config8 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 29;
    static const unsigned data_transfer_in = 16;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 16;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned reuse_factor = 24;
    static const unsigned n_zeros = 476;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef config8_mult mult_config;
};
const ap_uint<config8::filt_height * config8::filt_width> config8::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_1_linear
struct linear_config9 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_1_lineartable_t table_t;
};

// q_activation_2
struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_2_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_2
struct config47 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned out_height = 30;
    static const unsigned out_width = 29;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_2
struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 24;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config11 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 29;
    static const unsigned data_transfer_in = 32;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned reuse_factor = 24;
    static const unsigned n_zeros = 1080;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef config11_mult mult_config;
};
const ap_uint<config11::filt_height * config11::filt_width> config11::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_2_linear
struct linear_config12 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_2_lineartable_t table_t;
};

// q_activation_3
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_3_table_t table_t;
};

// max_pooling2d_1
struct config14 : nnet::pooling2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef max_pooling2d_1_default_t accum_t;
};

// zp2d_q_conv2d_batchnorm_3
struct config48 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned data_transfer_out = 32;
    static const unsigned n_chan = 32;
    static const unsigned out_height = 16;
    static const unsigned out_width = 15;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_3
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 96;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 15;
    static const unsigned data_transfer_in = 32;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned reuse_factor = 96;
    static const unsigned n_zeros = 5645;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    typedef config15_mult mult_config;
};
const ap_uint<config15::filt_height * config15::filt_width> config15::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_3_linear
struct linear_config16 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_3_lineartable_t table_t;
};

// q_activation_4
struct relu_config17 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_4_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_4
struct config49 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 16;
    static const unsigned out_width = 15;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_4
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 96;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config18 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 15;
    static const unsigned data_transfer_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned reuse_factor = 96;
    static const unsigned n_zeros = 10872;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    typedef config18_mult mult_config;
};
const ap_uint<config18::filt_height * config18::filt_width> config18::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_4_linear
struct linear_config19 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_4_lineartable_t table_t;
};

// q_activation_5
struct relu_config20 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_5_table_t table_t;
};

// max_pooling2d_2
struct config21 : nnet::pooling2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef max_pooling2d_2_default_t accum_t;
};

// zp2d_q_conv2d_batchnorm_5
struct config50 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 9;
    static const unsigned out_width = 8;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_5
struct config22_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias22_t bias_t;
    typedef weight22_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config22 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 8;
    static const unsigned data_transfer_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned reuse_factor = 288;
    static const unsigned n_zeros = 21053;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias22_t bias_t;
    typedef weight22_t weight_t;
    typedef config22_mult mult_config;
};
const ap_uint<config22::filt_height * config22::filt_width> config22::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_5_linear
struct linear_config23 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_5_lineartable_t table_t;
};

// q_activation_6
struct relu_config24 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_6_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_6
struct config51 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 9;
    static const unsigned out_width = 8;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_6
struct config25_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config25 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 8;
    static const unsigned data_transfer_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned reuse_factor = 288;
    static const unsigned n_zeros = 35112;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    typedef config25_mult mult_config;
};
const ap_uint<config25::filt_height * config25::filt_width> config25::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_6_linear
struct linear_config26 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_6_lineartable_t table_t;
};

// q_activation_7
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_7_table_t table_t;
};

// max_pooling2d_3
struct config28 : nnet::pooling2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef max_pooling2d_3_default_t accum_t;
};

// zp2d_q_conv2d_batchnorm_7
struct config52 : nnet::padding2d_config {
    static const unsigned in_height = 3;
    static const unsigned in_width = 3;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 5;
    static const unsigned out_width = 5;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_7
struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config29 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 5;
    static const unsigned in_width = 5;
    static const unsigned data_transfer_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned reuse_factor = 1152;
    static const unsigned n_zeros = 86149;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    typedef config29_mult mult_config;
};
const ap_uint<config29::filt_height * config29::filt_width> config29::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_7_linear
struct linear_config30 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_7_lineartable_t table_t;
};

// q_activation_8
struct relu_config31 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_8_table_t table_t;
};

// zp2d_q_conv2d_batchnorm_8
struct config53 : nnet::padding2d_config {
    static const unsigned in_height = 3;
    static const unsigned in_width = 3;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 5;
    static const unsigned out_width = 5;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_8
struct config32_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias32_t bias_t;
    typedef weight32_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config32 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 5;
    static const unsigned in_width = 5;
    static const unsigned data_transfer_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned reuse_factor = 1152;
    static const unsigned n_zeros = 317990;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias32_t bias_t;
    typedef weight32_t weight_t;
    typedef config32_mult mult_config;
};
const ap_uint<config32::filt_height * config32::filt_width> config32::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_8_linear
struct linear_config33 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_conv2d_batchnorm_8_lineartable_t table_t;
};

// q_activation_9
struct relu_config34 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_9_table_t table_t;
};

// q_dense_batchnorm
struct config36 : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 350041;
    static const unsigned n_nonzeros = 239783;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias36_t bias_t;
    typedef weight36_t weight_t;
    typedef layer36_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_batchnorm_linear
struct linear_config37 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_dense_batchnorm_lineartable_t table_t;
};

// q_activation_10
struct relu_config38 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_10_table_t table_t;
};

// q_dense_batchnorm_1
struct config39 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 7838;
    static const unsigned n_nonzeros = 57698;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias39_t bias_t;
    typedef weight39_t weight_t;
    typedef layer39_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_batchnorm_1_linear
struct linear_config40 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_dense_batchnorm_1_lineartable_t table_t;
};

// q_activation_11
struct relu_config41 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_11_table_t table_t;
};

// q_dense
struct config42 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias42_t bias_t;
    typedef weight42_t weight_t;
    typedef layer42_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_linear
struct linear_config43 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_dense_lineartable_t table_t;
};

// activation
struct relu_config44 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned data_transfer_out = 1;
    static const unsigned n_chan = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activationtable_t table_t;
};


#endif
