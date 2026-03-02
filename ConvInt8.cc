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

// Testcase4: is_optimize=true with 1x1 convolution
// Input: 1x1x1x4 (batch=1, h=1, w=1, in_channel=4)
// Kernel: 1x1, output_channel=2
// Output: 1x1x1x2
// This tests the optimized code path with matmul_func
TEST_F(ConvInt8Test, ConvInt8_1x1_optimize_true) {
  // Input data
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

  // For is_optimize=true:
  // unit_size = UP_ROUND(kernel_plane * in_channel, C4NUM) = UP_ROUND(4, 4) = 4
  // up_round_oc = UP_ROUND(out_channel, C8NUM) = UP_ROUND(2, 8) = 8
  int unit_size = 4;   // C4NUM aligned
  int up_round_oc = 8;  // C8NUM aligned
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Weight: [[1, 2, 3, 4], [5, 6, 7, 8]]
  // Packed in 4x16 major format for optimized path
  packed_weight[0] = 1;
  packed_weight[1] = 2;
  packed_weight[2] = 3;
  packed_weight[3] = 4;
  packed_weight[4] = 5;
  packed_weight[5] = 6;
  packed_weight[6] = 7;
  packed_weight[7] = 8;

  // Bias
  std::vector<int32_t> bias_data = {0, 0};

  // Filter zero point
  std::vector<int32_t> filter_zp = {0, 0};

  // Output buffer
  std::vector<int8_t> output_data(2, 0);

  // Temporary buffers
  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;  // tile_n * up_round_oc
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 1, 0);  // deep aligned to C4NUM
  std::vector<int32_t> input_sum(input_sum_size, 0);

  // Use MatMulInt8_4x16_r as the matmul function for optimized path
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  // Call ConvInt8 with is_optimize=true
  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0,  // task_id
           &conv_param, matmul_func, true);

  std::cout << "ConvInt8Test-ConvInt8_1x1_optimize_true output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Expected: [-64*1 + -32*2 + 0*3 + 32*4, -64*5 + -32*6 + 0*7 + 32*8]
  // = [0, -256] -> clipped to [0, -128]
  std::vector<int8_t> benchmark = {0, -128};

  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  std::cout << "Similarity: " << similarity << std::endl;
  ASSERT_GT(similarity, 0.99f);
}

// Testcase5: is_optimize=true with per-channel quantization
// Input: 1x2x2x2
// Kernel: 1x1, output_channel=8
// Output: 1x2x2x8
TEST_F(ConvInt8Test, ConvInt8_optimize_per_channel) {
  // Input: 1x2x2x2 in NHWC format
  std::vector<int8_t> input_data = {
    16, -16,   // (0,0)
    -32, 32,   // (0,1)
    48, -48,   // (1,0)
    -64, 64    // (1,1)
  };

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
  conv_param.output_channel_ = 8;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 4;

  // Per-channel quantization
  QuantArg input_quant_arg = {0.5f, 0};
  std::vector<QuantArg> filter_quant_args(8);
  for (int i = 0; i < 8; i++) {
    filter_quant_args[i] = {0.25f, 0};
  }
  QuantArg output_quant_arg = {1.0f, 0};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_args.data();
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 8;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

  std::vector<int32_t> quant_multiplier(8, 1073741824);
  std::vector<int32_t> left_shift(8, 0);
  std::vector<int32_t> right_shift(8, 0);
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier.data();
  conv_param.conv_quant_arg_.left_shift_ = left_shift.data();
  conv_param.conv_quant_arg_.right_shift_ = right_shift.data();

  int32_t act_min = -128;
  int32_t act_max = 127;
  conv_param.conv_quant_arg_.out_act_min_ = &act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &act_max;

  // For is_optimize=true with 8 output channels
  int unit_size = 4;   // UP_ROUND(2, 4)
  int up_round_oc = 8;  // UP_ROUND(8, 8)
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Simple weight matrix: alternating 1 and -1
  for (int oc = 0; oc < 8; oc++) {
    for (int ic = 0; ic < 2; ic++) {
      packed_weight[oc * unit_size + ic] = (oc % 2 == 0) ? 1 : -1;
    }
  }

  std::vector<int32_t> bias_data(8, 0);
  std::vector<int32_t> filter_zp(8, 0);
  std::vector<int8_t> output_data(32, 0);  // 2*2*8

  int kernel_plane = 1;
  int input_sum_size = 4 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 4, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 4, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, &conv_param, matmul_func, true);

  std::cout << "ConvInt8Test-ConvInt8_optimize_per_channel output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Verify output is not all zeros (basic sanity check)
  bool has_nonzero = false;
  for (auto val : output_data) {
    if (val != 0) {
      has_nonzero = true;
      break;
    }
  }
  ASSERT_TRUE(has_nonzero);
}

// Testcase6: is_optimize=true with larger input (2x2 spatial, 4 channels)
// Input: 1x2x2x4
// Kernel: 1x1, output_channel=4
// Output: 1x2x2x4
TEST_F(ConvInt8Test, ConvInt8_optimize_2x2_4ch) {
  // Input: 1x2x2x4 in NHWC format
  std::vector<int8_t> input_data = {
    1, 2, 3, 4,     // (0,0)
    5, 6, 7, 8,     // (0,1)
    9, 10, 11, 12,  // (1,0)
    13, 14, 15, 16  // (1,1)
  };

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
  conv_param.input_channel_ = 4;
  conv_param.output_h_ = 2;
  conv_param.output_w_ = 2;
  conv_param.output_channel_ = 4;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 4;

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

  // For is_optimize=true with 4 input and output channels
  int unit_size = 4;   // UP_ROUND(4, 4)
  int up_round_oc = 8;  // UP_ROUND(4, 8)
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Identity matrix weight
  for (int i = 0; i < 4; i++) {
    packed_weight[i * unit_size + i] = 1;
  }

  std::vector<int32_t> bias_data = {0, 0, 0, 0};
  std::vector<int32_t> filter_zp = {0};
  std::vector<int8_t> output_data(16, 0);  // 2*2*4

  int kernel_plane = 1;
  int input_sum_size = 4 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 4, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 4, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, &conv_param, matmul_func, true);

  std::cout << "ConvInt8Test-ConvInt8_optimize_2x2_4ch output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Expected: input_data unchanged (identity matrix)
  std::vector<int8_t> benchmark = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  std::cout << "Similarity: " << similarity << std::endl;
  ASSERT_GT(similarity, 0.99f);
}

// Testcase7: is_optimize=true with 3x3 kernel
// Input: 1x4x4x1
// Kernel: 3x3, output_channel=2
// Output: 1x2x2x2
TEST_F(ConvInt8Test, ConvInt8_optimize_3x3_kernel) {
  // Input: 1x4x4x1
  std::vector<int8_t> input_data = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
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
  conv_param.input_h_ = 4;
  conv_param.input_w_ = 4;
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 2;
  conv_param.output_w_ = 2;
  conv_param.output_channel_ = 2;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 4;

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

  // For is_optimize=true with 3x3 kernel
  int unit_size = 4;   // UP_ROUND(9, 4)
  int up_round_oc = 8;  // UP_ROUND(2, 8)
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Simple 3x3 kernel: all ones for first output channel, alternating for second
  for (int i = 0; i < 9; i++) {
    packed_weight[i] = 1;
    packed_weight[unit_size + i] = (i % 2 == 0) ? 1 : -1;
  }

  std::vector<int32_t> bias_data = {0, 0};
  std::vector<int32_t> filter_zp = {0};
  std::vector<int8_t> output_data(8, 0);  // 2*2*2

  int kernel_plane = 9;
  int input_sum_size = 4 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 4, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 4, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, &conv_param, matmul_func, true);

  std::cout << "ConvInt8Test-ConvInt8_optimize_3x3_kernel output:\n";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  // Verify output has valid values
  bool has_valid_output = false;
  for (auto val : output_data) {
    if (val != 0 || val == -128) {
      has_valid_output = true;
      break;
    }
  }
  ASSERT_TRUE(has_valid_output);
}

}  // namespace mindspore
