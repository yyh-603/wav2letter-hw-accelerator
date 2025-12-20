/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "cfu.h"
#include "perf.h"


namespace tflite {
namespace reference_integer_ops {

// Copyright (c) 2023-2024 Chung-Yi Chen (Yeecy)
// Faster Implementation of RDBPOT and SRDHM
inline int32_t RDBPOT(const int32_t x, const int32_t exp) {
  // const int32_t mask = (1 << exp) - 1;
  // const int32_t remainder = x & mask;
  // const int32_t threshold = (mask >> 1) + ((x >> 31) & 1);
  // return (x >> exp) + (remainder > threshold);
  return cfu_op1(1, static_cast<uint32_t>(x), static_cast<uint32_t>(exp));
}

inline int32_t SRDHM(const int32_t a, const int32_t b) {
  // const bool overflow = (a == INT32_MIN) && (b == INT32_MIN);
  // const int64_t ab_64 = (int64_t)a * (int64_t)b;
  // const int32_t nudge = ((ab_64 >> 63) & 1) ? 0xc0000001 : 0x40000000;
  // const int64_t ab_64_nudge = ab_64 + nudge;
  // return overflow ? 0x7fffffff :
  //   (ab_64_nudge >> 63) & 1 ? -(-ab_64_nudge >> 31) : ab_64_nudge >> 31;
  return cfu_op1(2, static_cast<uint32_t>(a), static_cast<uint32_t>(b));
}

inline int32_t myMultiplyByQuantizedMultiplier(int32_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {
  if (shift < 0) {
    return RDBPOT(SRDHM(x, quantized_multiplier), -shift);
  }
  else {
    return SRDHM(x << shift, quantized_multiplier);
  }
}

inline int8_t myRequant(
  int32_t acc, const int32_t output_multiplier, 
  const int32_t output_shift, const int32_t output_offset, 
  const int32_t output_activation_min, const int32_t output_activation_max) {
  
  acc = myMultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
  acc += output_offset;
  acc = std::max(acc, output_activation_min);
  acc = std::min(acc, output_activation_max);
  return static_cast<int8_t>(acc);
}

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  
  perf_enable_counter(7);
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  // const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);


  // var declaration
  TFLITE_DCHECK_EQ(groups, 1);

  const uint32_t K = filter_height * filter_width * filter_input_depth;
  const uint32_t P = output_height * output_width;

  constexpr uint32_t KMax = 8192;
  constexpr uint32_t PMax = 256;
  constexpr uint32_t KernelMax = 2048;

  int8_t im2col[PMax][KMax];
  int8_t kernel[KMax][KernelMax];
  int32_t result_arr[PMax][KernelMax];

  // im2col
  perf_enable_counter(0);
  for (int out_y = 0; out_y < output_height; ++out_y){
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int p = out_y * output_width + out_x;
      int k_idx = 0;
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = (out_y * stride_height) - pad_height + dilation_height_factor * filter_y;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = (out_x * stride_width) - pad_width + dilation_width_factor * filter_x;
          for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
            int32_t val = -input_offset;
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
              val = static_cast<int32_t>(input_data[Offset(input_shape, 0, in_y, in_x, in_channel)]);
            }
            im2col[p][k_idx] = static_cast<int8_t>(val);
            ++k_idx;
          }
        }
      }
    }
  }
  // perf_disable_counter(0);
 
  // kernel
  // perf_enable_counter(1);
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int k_idx = 0;
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
          kernel[k_idx][out_channel] = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
          ++k_idx;
        }
      }
    }
  }
  // perf_disable_counter(1);


  // GEMM
  // for (uint32_t p = 0; p < P; ++p){
  //   for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
  //     int32_t acc = 0;
  //     for (uint32_t k = 0; k < K; ++k) {
  //       acc += static_cast<int32_t>(int16_t(im2col[p][k]) + input_offset) * 
  //              static_cast<int32_t>(kernel[k][out_channel]);
               
  //     }
  //     if (bias_data)
  //       acc += bias_data[out_channel];
  //     result_arr_sw[p][out_channel] = acc;
  //   }
  // }

  // CFU GEMM (M * K @ K * N)
  const uint32_t M = P;
  const uint32_t N = output_depth;
  constexpr uint32_t T = 64;

  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j)
      result_arr[i][j] = 0;
  }

  for (uint32_t m0 = 0; m0 < M; m0 += T) {
    const uint32_t M_tile = std::min(T, M - m0);
    for (uint32_t n0 = 0; n0 < N; n0 += T) {
      const uint32_t N_tile = std::min(T, N - n0);
      for (uint32_t k0 = 0; k0 < K; k0 += T) {
        const uint32_t K_tile = std::min(T, K - k0);
        int A_row_cnt = 0;

        // perf_enable_counter(2);
        for (uint32_t ii = 0; ii < M_tile; ii += 4) {
          for (uint32_t jj = 0; jj < K_tile; ++jj) {
            const uint32_t i = m0 + ii;
            const uint32_t j = k0 + jj;
  
            const int8_t a0 = im2col[i + 0][j];
            const int8_t a1 = (ii + 1 < M_tile) ? im2col[i + 1][j] : static_cast<int8_t>(-input_offset);
            const int8_t a2 = (ii + 2 < M_tile) ? im2col[i + 2][j] : static_cast<int8_t>(-input_offset);
            const int8_t a3 = (ii + 3 < M_tile) ? im2col[i + 3][j] : static_cast<int8_t>(-input_offset);
  
            const uint32_t data0 = (static_cast<uint32_t>(static_cast<uint8_t>(a0)) << 24);
            const uint32_t data1 = (static_cast<uint32_t>(static_cast<uint8_t>(a1)) << 16);
            const uint32_t data2 = (static_cast<uint32_t>(static_cast<uint8_t>(a2)) <<  8);
            const uint32_t data3 = (static_cast<uint32_t>(static_cast<uint8_t>(a3)));
  
            const uint32_t data = data0 | data1 | data2 | data3;
            cfu_op0(1, A_row_cnt, data);
            ++A_row_cnt;
          }
        }
        // perf_disable_counter(2);

        // perf_enable_counter(3);
        int B_row_cnt = 0;
        for (uint32_t jj = 0; jj < N_tile; jj += 4) {
          for (uint32_t ii = 0; ii < K_tile; ++ii) {
            const uint32_t i = k0 + ii;
            const uint32_t j = n0 + jj;

            const int8_t b0 = kernel[i][j];
            const int8_t b1 = (jj + 1 < N_tile) ? kernel[i][j + 1] : 0;
            const int8_t b2 = (jj + 2 < N_tile) ? kernel[i][j + 2] : 0;
            const int8_t b3 = (jj + 3 < N_tile) ? kernel[i][j + 3] : 0;

            const uint32_t data0 = (static_cast<uint32_t>(static_cast<uint8_t>(b0)) << 24);
            const uint32_t data1 = (static_cast<uint32_t>(static_cast<uint8_t>(b1)) << 16);
            const uint32_t data2 = (static_cast<uint32_t>(static_cast<uint8_t>(b2)) <<  8);
            const uint32_t data3 = (static_cast<uint32_t>(static_cast<uint8_t>(b3)));

            const uint32_t data = data0 | data1 | data2 | data3;
            cfu_op0(2, B_row_cnt, data);
            ++B_row_cnt;
          }
        }
        // perf_disable_counter(3);

        // perf_enable_counter(4);
        const uint32_t dim = (static_cast<uint32_t>(K_tile) << 16) |
                             (static_cast<uint32_t>(M_tile) << 8)  |
                             static_cast<uint32_t>(N_tile);
        cfu_op0(3, dim, static_cast<uint32_t>(input_offset));
        // perf_disable_counter(4);

      }
      // perf_enable_counter(5);
      uint32_t C_row_cnt = 0;
      for (uint32_t jj = 0; jj < N_tile; jj += 4) {
        for (uint32_t ii = 0; ii < M_tile; ++ii) {
          const uint32_t i = m0 + ii;
          const uint32_t j = n0 + jj;

          result_arr[i][j] = static_cast<int32_t>(cfu_op0(4, C_row_cnt, 0));
          if (jj + 1 < N_tile) {
            result_arr[i][j + 1] = static_cast<int32_t>(cfu_op0(4, C_row_cnt, 1));
          }
          if (jj + 2 < N_tile) {
            result_arr[i][j + 2] = static_cast<int32_t>(cfu_op0(4, C_row_cnt, 2)); 
          }
          if (jj + 3 < N_tile) {
            result_arr[i][j + 3] = static_cast<int32_t>(cfu_op0(4, C_row_cnt, 3));
          }
          ++C_row_cnt;
        }
      }
      for (uint32_t i = 0; i < 1024; ++i)
        cfu_op0(5, i, 0);
      // perf_disable_counter(5);
    }
  }

  // perf_enable_counter(6);
  if (bias_data){
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < N; ++j) {
        result_arr[i][j] += bias_data[j];
      }
    }
  }

  // result to output tensor
  for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int p = out_y * output_width + out_x;
      for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        int32_t acc = result_arr[p][out_channel];
        // acc = MultiplyByQuantizedMultiplier(
        //   acc, output_multiplier[out_channel], output_shift[out_channel]);
        // acc += output_offset;
        // acc = std::max(acc, output_activation_min);
        // acc = std::min(acc, output_activation_max);
        // int32_t my_acc = myRequant(acc, output_multiplier[out_channel],
        //               output_shift[out_channel], output_offset, 
        //               output_activation_min, output_activation_max);
        // if (acc != my_acc) {
        //   printf("(acc, my_acc) = (%ld, %ld) (multiplier, shift, offset) = (%ld, %ld, %ld)\n", acc, my_acc, output_multiplier[out_channel], output_shift[out_channel], output_offset);
        // }
        // output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] = acc;
        output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] =
            myRequant(acc, output_multiplier[out_channel],
                      output_shift[out_channel], output_offset, 
                      output_activation_min, output_activation_max);
      }
    }
  }
  // perf_disable_counter(6);
  perf_disable_counter(7);

}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
