// Testcase1: Simple 1x1 convolution
// Input: 1x1x1x4 (batch=1, h=1, w=1, in_channel=4)
// Kernel: 1x1, output_channel=2
// Output: 1x1x1x2
TEST_F(ConvInt8Test, ConvInt8_1x1_simple) {
  // Input data - use heap allocation to avoid ASAN stack overflow
  std::vector<int8_t> input_data = {-64, -32, 0, 32};

  // Convolution parameters
  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));
  conv_param.kernel_h_ = 1;
  conv_param.kernel_w_ = 1;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.input_batch_ = 1;
  conv_param.input_h_ = 1;
  conv_param.input_w_ = 1;
  conv_param.input_channel_ = 4;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 2;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  // Quantization parameters (per-layer)
  QuantArg input_quant_arg = {0.5f, 0};
  QuantArg filter_quant_arg = {0.25f, 0};
  QuantArg output_quant_arg = {1.0f, 0};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = &filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  // Quantization multipliers
  int32_t quant_multiplier = 1073741824;
  int32_t left_shift = 0;
  int32_t right_shift = 0;
  conv_param.conv_quant_arg_.quant_multiplier_ = &quant_multiplier;
  conv_param.conv_quant_arg_.left_shift_ = &left_shift;
  conv_param.conv_quant_arg_.right_shift_ = &right_shift;

  // Activation min/max
  int32_t act_min = -128;
  int32_t act_max = 127;
  conv_param.conv_quant_arg_.out_act_min_ = &act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &act_max;

  // Packed weight for 1x1 conv with is_optimize=false
  // Unit size = UP_ROUND(kernel_plane * in_channel, 16) = UP_ROUND(4, 16) = 16
  // Up rounded OC = UP_ROUND(2, 4) = 4
  int unit_size = 16;
  int up_round_oc = 4;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Simple weight: [[1, 2, 3, 4], [5, 6, 7, 8]]
  packed_weight[0] = 1;
  packed_weight[1] = 2;
  packed_weight[2] = 3;
  packed_weight[3] = 4;
  packed_weight[16] = 5;
  packed_weight[17] = 6;
  packed_weight[18] = 7;
  packed_weight[19] = 8;

  // Bias
  std::vector<int32_t> bias_data = {0, 0};

  // Filter zero point
  std::vector<int32_t> filter_zp = {0, 0};

  // Output buffer
  std::vector<int8_t> output_data(2, 0);

  // Temporary buffers - use heap to avoid stack overflow
  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 16 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  // Call ConvInt8 with is_optimize=false
  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0,  // task_id
           &conv_param, nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_1x1_simple output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Expected: [-64*1 + -32*2 + 0*3 + 32*4, -64*5 + -32*6 + 0*7 + 32*8]
  // = [-64 - 64 + 0 + 128, -320 - 192 + 0 + 256]
  // = [0, -256] -> clipped to [0, -128]
  std::vector<int8_t> benchmark = {0, -128};

  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  std::cout << "Similarity: " << similarity << std::endl;
  ASSERT_GT(similarity, 0.99f);
}

// Testcase2: 1x1 convolution with 2x2 spatial input
// Input: 1x2x2x2 (batch=1, h=2, w=2, in_channel=2)
// Kernel: 1x1, output_channel=2
// Output: 1x2x2x2
TEST_F(ConvInt8Test, ConvInt8_1x1_2x2) {
  // Input: 1x2x2x2 in NHWC format
  std::vector<int8_t> input_data = {
    -64, 32,   // (0,0)
    -32, 0,    // (0,1)
    16, -16,   // (1,0)
    48, -48    // (1,1)
  };

  // Convolution parameters
  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));
  conv_param.kernel_h_ = 1;
  conv_param.kernel_w_ = 1;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.input_batch_ = 1;
  conv_param.input_h_ = 2;
  conv_param.input_w_ = 2;
  conv_param.input_channel_ = 2;
  conv_param.output_h_ = 2;
  conv_param.output_w_ = 2;
  conv_param.output_channel_ = 2;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 4;  // output_h * output_w

  // Quantization parameters
  QuantArg input_quant_arg = {0.5f, 0};
  QuantArg filter_quant_arg = {0.25f, 0};
  QuantArg output_quant_arg = {1.0f, 0};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = &filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int32_t quant_multiplier = 1073741824;
  int32_t left_shift = 0;
  int32_t right_shift = 0;
  conv_param.conv_quant_arg_.quant_multiplier_ = &quant_multiplier;
  conv_param.conv_quant_arg_.left_shift_ = &left_shift;
  conv_param.conv_quant_arg_.right_shift_ = &right_shift;

  int32_t act_min = -128;
  int32_t act_max = 127;
  conv_param.conv_quant_arg_.out_act_min_ = &act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &act_max;

  // Packed weight
  int unit_size = 16;  // UP_ROUND(1*1*2, 16)
  int up_round_oc = 4;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Weight: [[1, -1], [2, -2]]
  packed_weight[0] = 1;
  packed_weight[1] = -1;
  packed_weight[16] = 2;
  packed_weight[17] = -2;

  std::vector<int32_t> bias_data = {0, 0};
  std::vector<int32_t> filter_zp = {0, 0};
  std::vector<int8_t> output_data(8, 0);

  int kernel_plane = 1;
  int input_sum_size = 4 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 4, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 16 * 4, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, &conv_param, nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_1x1_2x2 output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Expected: For each pixel, [c0*1 + c1*(-1), c0*2 + c1*(-2)]
  // (0,0): [-64*1 + 32*(-1), -64*2 + 32*(-2)] = [-96, -192] -> [-96, -127]
  // (0,1): [-32*1 + 0*(-1), -32*2 + 0*(-2)] = [-32, -64]
  // (1,0): [16*1 + (-16)*(-1), 16*2 + (-16)*(-2)] = [32, 64]
  // (1,1): [48*1 + (-48)*(-1), 48*2 + (-48)*(-2)] = [96, 192] -> [96, 127]
  std::vector<int8_t> benchmark = {-96, -127, -32, -64, 32, 64, 96, 127};

  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  std::cout << "Similarity: " << similarity << std::endl;
  ASSERT_GT(similarity, 0.99f);
}

// Testcase3: Test with 3x3 kernel
// Input: 1x3x3x1
// Kernel: 3x3, output_channel=1
TEST_F(ConvInt8Test, ConvInt8_3x3_kernel) {
  // Input: 1x3x3x1
  std::vector<int8_t> input_data = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));
  conv_param.kernel_h_ = 3;
  conv_param.kernel_w_ = 3;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.input_batch_ = 1;
  conv_param.input_h_ = 3;
  conv_param.input_w_ = 3;
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  // Quantization parameters
  QuantArg input_quant_arg = {1.0f, 0};
  QuantArg filter_quant_arg = {1.0f, 0};
  QuantArg output_quant_arg = {1.0f, 0};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = &filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int32_t quant_multiplier = 1073741824;
  int32_t left_shift = 0;
  int32_t right_shift = 0;
  conv_param.conv_quant_arg_.quant_multiplier_ = &quant_multiplier;
  conv_param.conv_quant_arg_.left_shift_ = &left_shift;
  conv_param.conv_quant_arg_.right_shift_ = &right_shift;

  int32_t act_min = -128;
  int32_t act_max = 127;
  conv_param.conv_quant_arg_.out_act_min_ = &act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &act_max;

  // Packed weight: 3x3 kernel, 1 input channel -> unit_size = UP_ROUND(9*1, 16) = 16
  int unit_size = 16;
  int up_round_oc = 4;  // UP_ROUND(1, 4)
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Weight: all ones
  for (int i = 0; i < 9; i++) {
    packed_weight[i] = 1;
  }

  std::vector<int32_t> bias_data = {0};
  std::vector<int32_t> filter_zp = {0};
  std::vector<int8_t> output_data(1, 0);

  int kernel_plane = 9;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 16 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, &conv_param, nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_3x3_kernel output: " << static_cast<int32_t>(output_data[0]) << std::endl;

  // Expected: sum of all input values = 1+2+3+4+5+6+7+8+9 = 45
  std::vector<int8_t> benchmark = {45};

  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  std::cout << "Similarity: " << similarity << std::endl;
  ASSERT_GT(similarity, 0.99f);
}