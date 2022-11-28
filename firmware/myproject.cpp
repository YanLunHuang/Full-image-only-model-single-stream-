//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> em_endcap[4],
    hls::stream<result_t> layer44_out[1],
	weight18_t w18[36864],
	weight22_t w22[73728],
	weight25_t w25[147456],
	weight29_t w29[294912],
	weight32_t w32[589824],
	weight36_t w36[589824],
	weight39_t w39[65536]
) {

    #pragma HLS INTERFACE axis port=em_endcap,layer44_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight4_t, 1600>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight8_t, 4608>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 9216>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight15_t, 18432>(w15, "w15.txt");
        nnet::load_weights_from_txt<bias15_t, 64>(b15, "b15.txt");
        //nnet::load_weights_from_txt<weight18_t, 36864>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 64>(b18, "b18.txt");
        //nnet::load_weights_from_txt<weight22_t, 73728>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 128>(b22, "b22.txt");
        //nnet::load_weights_from_txt<weight25_t, 147456>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 128>(b25, "b25.txt");
        //nnet::load_weights_from_txt<weight29_t, 294912>(w29, "w29.txt");
        nnet::load_weights_from_txt<bias29_t, 256>(b29, "b29.txt");
        //nnet::load_weights_from_txt<weight32_t, 589824>(w32, "w32.txt");
        nnet::load_weights_from_txt<bias32_t, 256>(b32, "b32.txt");
        //nnet::load_weights_from_txt<weight36_t, 589824>(w36, "w36.txt");
        nnet::load_weights_from_txt<bias36_t, 256>(b36, "b36.txt");
        //nnet::load_weights_from_txt<weight39_t, 65536>(w39, "w39.txt");
        nnet::load_weights_from_txt<bias39_t, 256>(b39, "b39.txt");
        nnet::load_weights_from_txt<weight42_t, 256>(w42, "w42.txt");
        nnet::load_weights_from_txt<bias42_t, 1>(b42, "b42.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out[4];
    #pragma HLS STREAM variable=layer2_out depth=3080
	 #pragma HLS RESOURCE variable=layer2_out core=FIFO_LUTRAM
    nnet::resize_nearest_switch<input_t, config2>(em_endcap, layer2_out); // up_sampling2d

    hls::stream<layer3_t> layer3_out[4];
    #pragma HLS STREAM variable=layer3_out depth=3080
	#pragma HLS RESOURCE variable=layer3_out core=FIFO_LUTRAM
    nnet::linear_switch<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // q_activation

    hls::stream<layer45_t> layer45_out[4];
    #pragma HLS STREAM variable=layer45_out depth=3540
    nnet::zeropad2d_cl_switch<layer3_t, layer45_t, config45>(layer3_out, layer45_out); // zp2d_q_conv2d_batchnorm

    hls::stream<layer4_t> layer4_out[16];
    #pragma HLS STREAM variable=layer4_out depth=3080
    nnet::conv_2d_cl_switch<layer45_t, layer4_t, config4>(layer45_out, layer4_out, w4, b4); // q_conv2d_batchnorm

    hls::stream<layer5_t> layer5_out[16];
    #pragma HLS STREAM variable=layer5_out depth=3080
	#pragma HLS RESOURCE variable=layer5_out core=FIFO_LUTRAM
    nnet::linear_switch<layer4_t, layer5_t, linear_config5>(layer4_out, layer5_out); // q_conv2d_batchnorm_linear

    hls::stream<layer6_t> layer6_out[16];
    #pragma HLS STREAM variable=layer6_out depth=3080
	#pragma HLS RESOURCE variable=layer6_out core=FIFO_LUTRAM
    nnet::relu_switch<layer5_t, layer6_t, relu_config6>(layer5_out, layer6_out); // q_activation_1

    hls::stream<layer7_t> layer7_out[16];
    #pragma HLS STREAM variable=layer7_out depth=756
	#pragma HLS RESOURCE variable=layer7_out core=FIFO_LUTRAM
    nnet::pooling2d_cl_switch<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // max_pooling2d

    hls::stream<layer46_t> layer46_out[16];
    #pragma HLS STREAM variable=layer46_out depth=870
    nnet::zeropad2d_cl_switch<layer7_t, layer46_t, config46>(layer7_out, layer46_out); // zp2d_q_conv2d_batchnorm_1

    hls::stream<layer8_t> layer8_out[32];
    #pragma HLS STREAM variable=layer8_out depth=756
    nnet::conv_2d_cl_switch<layer46_t, layer8_t, config8>(layer46_out, layer8_out, w8, b8); // q_conv2d_batchnorm_1

    hls::stream<layer9_t> layer9_out[32];
    #pragma HLS STREAM variable=layer9_out depth=756
    nnet::linear_switch<layer8_t, layer9_t, linear_config9>(layer8_out, layer9_out); // q_conv2d_batchnorm_1_linear

    hls::stream<layer10_t> layer10_out[32];
    #pragma HLS STREAM variable=layer10_out depth=756
    nnet::relu_switch<layer9_t, layer10_t, relu_config10>(layer9_out, layer10_out); // q_activation_2

    hls::stream<layer47_t> layer47_out[32];
    #pragma HLS STREAM variable=layer47_out depth=870
    nnet::zeropad2d_cl_switch<layer10_t, layer47_t, config47>(layer10_out, layer47_out); // zp2d_q_conv2d_batchnorm_2

    hls::stream<layer11_t> layer11_out[32];
    #pragma HLS STREAM variable=layer11_out depth=756
    nnet::conv_2d_cl_switch<layer47_t, layer11_t, config11>(layer47_out, layer11_out, w11, b11); // q_conv2d_batchnorm_2

    hls::stream<layer12_t> layer12_out[32];
    #pragma HLS STREAM variable=layer12_out depth=756
    nnet::linear_switch<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // q_conv2d_batchnorm_2_linear

    hls::stream<layer13_t> layer13_out[32];
    #pragma HLS STREAM variable=layer13_out depth=756
    nnet::relu_switch<layer12_t, layer13_t, relu_config13>(layer12_out, layer13_out); // q_activation_3

    hls::stream<layer14_t> layer14_out[32];
    #pragma HLS STREAM variable=layer14_out depth=182
    nnet::pooling2d_cl_switch<layer13_t, layer14_t, config14>(layer13_out, layer14_out); // max_pooling2d_1

    hls::stream<layer48_t> layer48_out[32];
    #pragma HLS STREAM variable=layer48_out depth=240
    nnet::zeropad2d_cl_switch<layer14_t, layer48_t, config48>(layer14_out, layer48_out); // zp2d_q_conv2d_batchnorm_3

    hls::stream<layer15_t> layer15_out[1];
    #pragma HLS STREAM variable=layer15_out depth=182
    nnet::conv_2d_cl_switch<layer48_t, layer15_t, config15>(layer48_out, layer15_out, w15, b15); // q_conv2d_batchnorm_3

    hls::stream<layer16_t> layer16_out[1];
    #pragma HLS STREAM variable=layer16_out depth=182
	#pragma HLS RESOURCE variable=layer16_out core=FIFO_LUTRAM
    nnet::linear_switch<layer15_t, layer16_t, linear_config16>(layer15_out, layer16_out); // q_conv2d_batchnorm_3_linear

    hls::stream<layer17_t> layer17_out[1];
    #pragma HLS STREAM variable=layer17_out depth=182
	#pragma HLS RESOURCE variable=layer17_out core=FIFO_LUTRAM
    nnet::relu_switch<layer16_t, layer17_t, relu_config17>(layer16_out, layer17_out); // q_activation_4

    hls::stream<layer49_t> layer49_out[1];
    #pragma HLS STREAM variable=layer49_out depth=240
    nnet::zeropad2d_cl_switch<layer17_t, layer49_t, config49>(layer17_out, layer49_out); // zp2d_q_conv2d_batchnorm_4

    hls::stream<layer18_t> layer18_out[1];
    #pragma HLS STREAM variable=layer18_out depth=182
    nnet::conv_2d_cl_switch<layer49_t, layer18_t, config18>(layer49_out, layer18_out, w18, b18); // q_conv2d_batchnorm_4

    hls::stream<layer19_t> layer19_out[1];
    #pragma HLS STREAM variable=layer19_out depth=182
	#pragma HLS RESOURCE variable=layer19_out core=FIFO_LUTRAM
    nnet::linear_switch<layer18_t, layer19_t, linear_config19>(layer18_out, layer19_out); // q_conv2d_batchnorm_4_linear

    hls::stream<layer20_t> layer20_out[1];
    #pragma HLS STREAM variable=layer20_out depth=182
	#pragma HLS RESOURCE variable=layer20_out core=FIFO_LUTRAM
    nnet::relu_switch<layer19_t, layer20_t, relu_config20>(layer19_out, layer20_out); // q_activation_5

    hls::stream<layer21_t> layer21_out[1];
    #pragma HLS STREAM variable=layer21_out depth=42
	#pragma HLS RESOURCE variable=layer21_out core=FIFO_LUTRAM
    nnet::pooling2d_cl_switch<layer20_t, layer21_t, config21>(layer20_out, layer21_out); // max_pooling2d_2

    hls::stream<layer50_t> layer50_out[1];
    #pragma HLS STREAM variable=layer50_out depth=72
    nnet::zeropad2d_cl_switch<layer21_t, layer50_t, config50>(layer21_out, layer50_out); // zp2d_q_conv2d_batchnorm_5

    hls::stream<layer22_t> layer22_out[1];
    #pragma HLS STREAM variable=layer22_out depth=42
    nnet::conv_2d_cl_switch<layer50_t, layer22_t, config22>(layer50_out, layer22_out, w22, b22); // q_conv2d_batchnorm_5

    hls::stream<layer23_t> layer23_out[1];
    #pragma HLS STREAM variable=layer23_out depth=42
	#pragma HLS RESOURCE variable=layer23_out core=FIFO_LUTRAM
    nnet::linear_switch<layer22_t, layer23_t, linear_config23>(layer22_out, layer23_out); // q_conv2d_batchnorm_5_linear

    hls::stream<layer24_t> layer24_out[1];
    #pragma HLS STREAM variable=layer24_out depth=42
	#pragma HLS RESOURCE variable=layer24_out core=FIFO_LUTRAM
    nnet::relu_switch<layer23_t, layer24_t, relu_config24>(layer23_out, layer24_out); // q_activation_6

    hls::stream<layer51_t> layer51_out[1];
    #pragma HLS STREAM variable=layer51_out depth=72
    nnet::zeropad2d_cl_switch<layer24_t, layer51_t, config51>(layer24_out, layer51_out); // zp2d_q_conv2d_batchnorm_6

    hls::stream<layer25_t> layer25_out[1];
    #pragma HLS STREAM variable=layer25_out depth=42
    nnet::conv_2d_cl_switch<layer51_t, layer25_t, config25>(layer51_out, layer25_out, w25, b25); // q_conv2d_batchnorm_6

    hls::stream<layer26_t> layer26_out[1];
    #pragma HLS STREAM variable=layer26_out depth=42
	#pragma HLS RESOURCE variable=layer26_out core=FIFO_LUTRAM
    nnet::linear_switch<layer25_t, layer26_t, linear_config26>(layer25_out, layer26_out); // q_conv2d_batchnorm_6_linear

    hls::stream<layer27_t> layer27_out[1];
    #pragma HLS STREAM variable=layer27_out depth=42
	#pragma HLS RESOURCE variable=layer27_out core=FIFO_LUTRAM
    nnet::relu_switch<layer26_t, layer27_t, relu_config27>(layer26_out, layer27_out); // q_activation_7

    hls::stream<layer28_t> layer28_out[1];
    #pragma HLS STREAM variable=layer28_out depth=9
	#pragma HLS RESOURCE variable=layer28_out core=FIFO_LUTRAM
    nnet::pooling2d_cl_switch<layer27_t, layer28_t, config28>(layer27_out, layer28_out); // max_pooling2d_3

    hls::stream<layer52_t> layer52_out[1];
    #pragma HLS STREAM variable=layer52_out depth=25
    nnet::zeropad2d_cl_switch<layer28_t, layer52_t, config52>(layer28_out, layer52_out); // zp2d_q_conv2d_batchnorm_7

    hls::stream<layer29_t> layer29_out[1];
    #pragma HLS STREAM variable=layer29_out depth=9
    nnet::conv_2d_cl_switch<layer52_t, layer29_t, config29>(layer52_out, layer29_out, w29, b29); // q_conv2d_batchnorm_7

    hls::stream<layer30_t> layer30_out[1];
    #pragma HLS STREAM variable=layer30_out depth=9
	#pragma HLS RESOURCE variable=layer30_out core=FIFO_LUTRAM
    nnet::linear_switch<layer29_t, layer30_t, linear_config30>(layer29_out, layer30_out); // q_conv2d_batchnorm_7_linear

    hls::stream<layer31_t> layer31_out[1];
    #pragma HLS STREAM variable=layer31_out depth=9
	#pragma HLS RESOURCE variable=layer31_out core=FIFO_LUTRAM
    nnet::relu_switch<layer30_t, layer31_t, relu_config31>(layer30_out, layer31_out); // q_activation_8

    hls::stream<layer53_t> layer53_out[1];
    #pragma HLS STREAM variable=layer53_out depth=25
    nnet::zeropad2d_cl_switch<layer31_t, layer53_t, config53>(layer31_out, layer53_out); // zp2d_q_conv2d_batchnorm_8

    hls::stream<layer32_t> layer32_out[1];
    #pragma HLS STREAM variable=layer32_out depth=9
    nnet::conv_2d_cl_switch<layer53_t, layer32_t, config32>(layer53_out, layer32_out, w32, b32); // q_conv2d_batchnorm_8

    hls::stream<layer33_t> layer33_out[1];
    #pragma HLS STREAM variable=layer33_out depth=9
	#pragma HLS RESOURCE variable=layer33_out core=FIFO_LUTRAM
    nnet::linear_switch<layer32_t, layer33_t, linear_config33>(layer32_out, layer33_out); // q_conv2d_batchnorm_8_linear

    hls::stream<layer34_t> layer34_out[1];
    #pragma HLS STREAM variable=layer34_out depth=9
	#pragma HLS RESOURCE variable=layer34_out core=FIFO_LUTRAM
    nnet::relu_switch<layer33_t, layer34_t, relu_config34>(layer33_out, layer34_out); // q_activation_9

    hls::stream<layer36_t> layer36_out[1];
    #pragma HLS STREAM variable=layer36_out depth=1
	#pragma HLS RESOURCE variable=layer36_out core=FIFO_LUTRAM
    nnet::dense_ss<layer34_t, layer36_t, config36>(layer34_out, layer36_out, w36, b36); // q_dense_batchnorm

    hls::stream<layer37_t> layer37_out[1];
    #pragma HLS STREAM variable=layer37_out depth=1
	#pragma HLS RESOURCE variable=layer37_out core=FIFO_LUTRAM
    nnet::linear_switch<layer36_t, layer37_t, linear_config37>(layer36_out, layer37_out); // q_dense_batchnorm_linear

    hls::stream<layer38_t> layer38_out[1];
    #pragma HLS STREAM variable=layer38_out depth=1
	#pragma HLS RESOURCE variable=layer38_out core=FIFO_LUTRAM
    nnet::relu_switch<layer37_t, layer38_t, relu_config38>(layer37_out, layer38_out); // q_activation_10

    hls::stream<layer39_t> layer39_out[1];
    #pragma HLS STREAM variable=layer39_out depth=1
	#pragma HLS RESOURCE variable=layer39_out core=FIFO_LUTRAM
    nnet::dense_ss<layer38_t, layer39_t, config39>(layer38_out, layer39_out, w39, b39); // q_dense_batchnorm_1

    hls::stream<layer40_t> layer40_out[1];
    #pragma HLS STREAM variable=layer40_out depth=1
	#pragma HLS RESOURCE variable=layer40_out core=FIFO_LUTRAM
    nnet::linear_switch<layer39_t, layer40_t, linear_config40>(layer39_out, layer40_out); // q_dense_batchnorm_1_linear

    hls::stream<layer41_t> layer41_out[1];
    #pragma HLS STREAM variable=layer41_out depth=1
	#pragma HLS RESOURCE variable=layer41_out core=FIFO_LUTRAM
    nnet::relu_switch<layer40_t, layer41_t, relu_config41>(layer40_out, layer41_out); // q_activation_11

    hls::stream<layer42_t> layer42_out[1];
    #pragma HLS STREAM variable=layer42_out depth=1
	#pragma HLS RESOURCE variable=layer42_out core=FIFO_LUTRAM
    nnet::dense_ss<layer41_t, layer42_t, config42>(layer41_out, layer42_out, w42, b42); // q_dense

    hls::stream<layer43_t> layer43_out[1];
    #pragma HLS STREAM variable=layer43_out depth=1
	#pragma HLS RESOURCE variable=layer43_out core=FIFO_LUTRAM
    nnet::linear_switch<layer42_t, layer43_t, linear_config43>(layer42_out, layer43_out); // q_dense_linear

    nnet::relu_switch<layer43_t, result_t, relu_config44>(layer43_out, layer44_out); // activation
}
