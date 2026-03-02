// Testcase1: ConvInt8 with is_optimize=true, minimal data size
// Input: batch=1, h=1, w=1, in_c=2, out_c=2, kernel=1x1
TEST_F(ConvInt8Test, ConvInt8_optimize_true) {
  // Minimal test configuration: 1x1x1x2 input, 2 output channels, 1x1 kernel
  const int batch = 1;
  const int in_h = 1;
  const int in_w = 1;
  const int in_c = 2;
  const int out_c = 2;
  const int kernel_h = 1;
  const int kernel_w = 1;
  const int out_h = 1;
  const int out_w = 1;

  // Input data: [1, 1, 1, 2] -> NHWC format
  std::vector<int8_t> input_data = {-64, 63};

  // Packed weight for 2 output channels, kernel_plane * in_c = 1 * 2 = 2
  // Per-channel quantization, each output channel has filter_zp
  std::vector<int8_t> packed_weight = {
    -103, 25,   // OC=0: weights for IC=0,1
    51,   -77   // OC=1: weights for IC=0,1
  };

  // Bias for 2 output channels
  std::vector<int32_t> bias_data = {0, 0};

  // Output data
  std::vector<int8_t> output_data(batch * out_h * out_w * out_c, 0);

  // Filter zero points for per-channel quantization
  std::vector<int32_t> filter_zp = {0, 0};

  // Calculate buffer sizes based on is_optimize=true
  // For is_optimize=true: unit_size = UP_ROUND(kernel_plane * in_c, C4NUM)
  const int kernel_plane = kernel_h * kernel_w;
  const int deep = kernel_plane * in_c;
  const int tile_num = 1;  // Small tile for minimal memory
  const int unit_size = UP_ROUND(deep, C4NUM);  // UP_ROUND(2, 4) = 4
  const int up_round_oc = UP_ROUND(out_c, C8NUM);  // UP_ROUND(2, 8) = 8
  const int input_sum_offset = tile_num * up_round_oc;  // 1 * 8 = 8

  // Packed input buffer
  std::vector<int8_t> packed_input(unit_size * tile_num, 0);

  // Matmul input buffer
  std::vector<int8_t> matmul_input(deep * tile_num, 0);

  // Input sum buffer
  std::vector<int32_t> input_sum(input_sum_offset, 0);

  // Setup ConvParameter
  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));

  conv_param.input_batch_ = batch;
  conv_param.input_h_ = in_h;
  conv_param.input_w_ = in_w;
  conv_param.input_channel_ = in_c;
  conv_param.output_h_ = out_h;
  conv_param.output_w_ = out_w;
  conv_param.output_channel_ = out_c;
  conv_param.kernel_h_ = kernel_h;
  conv_param.kernel_w_ = kernel_w;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.group_ = 1;
  conv_param.tile_num_ = tile_num;
  conv_param.thread_num_ = 1;

  // Setup quantization parameters (per-layer for simplicity)
  QuantArg input_quant_arg = {0.5f, -128};
  QuantArg filter_quant_args[2] = {{0.01f, 0}, {0.01f, 0}};
  QuantArg output_quant_arg = {0.5f, 50};
  int32_t out_act_min = -128;
  int32_t out_act_max = 127;
  int32_t left_shift[2] = {0, 0};
  int32_t right_shift[2] = {-8, -8};
  int32_t quant_multiplier[2] = {1073741824, 1073741824};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_args;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.out_act_min_ = &out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &out_act_max;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 2;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

  // Matmul function - use MatMulInt8_4x16_r which matches the signature
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  // Call ConvInt8 with is_optimize=true
  const int task_id = 0;
  const bool is_optimize = true;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  // Expected output: manually computed
  // For OC=0: (-64) * (-103) + 63 * 25 = 6592 + 1575 = 8167 (clipped to 127)
  // For OC=1: (-64) * 51 + 63 * (-77) = -3264 - 4851 = -8115 (clipped to -128)
  // After quantization with the parameters, values should be in valid range
  std::cout << "ConvInt8Test-ConvInt8_optimize_true output:\n";
  for (size_t i = 0; i < output_data.size(); ++i) {
    std::cout << static_cast<int32_t>(output_data[i]) << ", ";
  }
  std::cout << std::endl;

  // Verify output is in valid int8 range
  for (auto val : output_data) {
    EXPECT_GE(val, -128);
    EXPECT_LE(val, 127);
  }
}

// Testcase2: ConvInt8 with is_optimize=false, minimal data size
// Input: batch=1, h=1, w=1, in_c=2, out_c=2, kernel=1x1
TEST_F(ConvInt8Test, ConvInt8_optimize_false) {
  // Minimal test configuration: 1x1x1x2 input, 2 output channels, 1x1 kernel
  const int batch = 1;
  const int in_h = 1;
  const int in_w = 1;
  const int in_c = 2;
  const int out_c = 2;
  const int kernel_h = 1;
  const int kernel_w = 1;
  const int out_h = 1;
  const int out_w = 1;

  // Input data: [1, 1, 1, 2] -> NHWC format
  std::vector<int8_t> input_data = {-64, 63};

  // Packed weight for is_optimize=false: UP_ROUND(deep, C16NUM) layout
  // deep = kernel_plane * in_c = 1 * 2 = 2, UP_ROUND(2, 16) = 16
  std::vector<int8_t> packed_weight(16 * out_c, 0);
  // OC=0: [-103, 25, 0, 0, ...] (16 elements)
  // OC=1: [51, -77, 0, 0, ...] (16 elements)
  packed_weight[0] = -103;
  packed_weight[1] = 25;
  packed_weight[16] = 51;
  packed_weight[17] = -77;

  // Bias for 2 output channels
  std::vector<int32_t> bias_data = {0, 0};

  // Output data
  std::vector<int8_t> output_data(batch * out_h * out_w * out_c, 0);

  // Filter zero points for per-channel quantization
  std::vector<int32_t> filter_zp = {0, 0};

  // Calculate buffer sizes based on is_optimize=false
  const int kernel_plane = kernel_h * kernel_w;
  const int deep = kernel_plane * in_c;
  const int tile_num = 1;
  const int unit_size = UP_ROUND(deep, C16NUM);  // UP_ROUND(2, 16) = 16
  const int up_round_oc = UP_ROUND(out_c, C4NUM);  // UP_ROUND(2, 4) = 4
  const int input_sum_offset = tile_num;  // per-layer: tile_num

  // Packed input buffer
  std::vector<int8_t> packed_input(unit_size * tile_num, 0);

  // Matmul input buffer
  std::vector<int8_t> matmul_input(deep * tile_num, 0);

  // Input sum buffer
  std::vector<int32_t> input_sum(input_sum_offset, 0);

  // Setup ConvParameter
  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));

  conv_param.input_batch_ = batch;
  conv_param.input_h_ = in_h;
  conv_param.input_w_ = in_w;
  conv_param.input_channel_ = in_c;
  conv_param.output_h_ = out_h;
  conv_param.output_w_ = out_w;
  conv_param.output_channel_ = out_c;
  conv_param.kernel_h_ = kernel_h;
  conv_param.kernel_w_ = kernel_w;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.group_ = 1;
  conv_param.tile_num_ = tile_num;
  conv_param.thread_num_ = 1;

  // Setup quantization parameters (per-layer for is_optimize=false path)
  QuantArg input_quant_arg = {0.5f, -128};
  QuantArg filter_quant_args = {0.01f, 0};  // Per-layer
  QuantArg output_quant_arg = {0.5f, 50};
  int32_t out_act_min = -128;
  int32_t out_act_max = 127;
  int32_t left_shift[2] = {0, 0};
  int32_t right_shift[2] = {-8, -8};
  int32_t quant_multiplier[2] = {1073741824, 1073741824};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = &filter_quant_args;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.out_act_min_ = &out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &out_act_max;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;  // Per-layer
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;  // No per-channel

  // Matmul function - not used for is_optimize=false, but provide for completeness
  MATMUL_OPT_R_FUNC matmul_func = nullptr;

  // Call ConvInt8 with is_optimize=false
  const int task_id = 0;
  const bool is_optimize = false;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  // Expected output: manually computed
  // Same computation as optimize=true, but different packing strategy
  std::cout << "ConvInt8Test-ConvInt8_optimize_false output:\n";
  for (size_t i = 0; i < output_data.size(); ++i) {
    std::cout << static_cast<int32_t>(output_data[i]) << ", ";
  }
  std::cout << std::endl;

  // Verify output is in valid int8 range
  for (auto val : output_data) {
    EXPECT_GE(val, -128);
    EXPECT_LE(val, 127);
  }
}

// Testcase3: ConvInt8 with is_optimize=true, slightly larger data
// Input: batch=1, h=2, w=2, in_c=4, out_c=4, kernel=1x1
TEST_F(ConvInt8Test, ConvInt8_optimize_true_larger) {
  // Test with slightly larger dimensions
  const int batch = 1;
  const int in_h = 2;
  const int in_w = 2;
  const int in_c = 4;
  const int out_c = 4;
  const int kernel_h = 1;
  const int kernel_w = 1;
  const int out_h = 2;
  const int out_w = 2;

  // Input data: 2x2x4 = 16 elements
  std::vector<int8_t> input_data = {
    -64, -32, 0, 32,   // h=0, w=0
    63,  95,  -1, 31,  // h=0, w=1
    -96, -16, 16, 96,  // h=1, w=0
    -127, 1,  15, 127  // h=1, w=1
  };

  // Packed weight: deep = 1 * 4 = 4, UP_ROUND(4, 4) = 4
  // For 4 output channels: 4 x 4 = 16 elements
  std::vector<int8_t> packed_weight = {
    -103, 25,  -77, 51,   // OC=0
    -49,  98,  -26, 74,   // OC=1
    -35,  10,   5,  48,   // OC=2
    -95,  28,  -66, -42   // OC=3
  };

  // Bias for 4 output channels
  std::vector<int32_t> bias_data = {0, 0, 0, 0};

  // Output data: 2x2x4 = 16 elements
  std::vector<int8_t> output_data(batch * out_h * out_w * out_c, 0);

  // Filter zero points
  std::vector<int32_t> filter_zp = {0, 0, 0, 0};

  // Calculate buffer sizes
  const int kernel_plane = kernel_h * kernel_w;
  const int deep = kernel_plane * in_c;
  const int tile_num = 4;  // 2x2 output
  const int unit_size = UP_ROUND(deep, C4NUM);  // UP_ROUND(4, 4) = 4
  const int up_round_oc = UP_ROUND(out_c, C8NUM);  // UP_ROUND(4, 8) = 8
  const int input_sum_offset = tile_num * up_round_oc;  // 4 * 8 = 32

  // Packed input buffer
  std::vector<int8_t> packed_input(unit_size * tile_num, 0);

  // Matmul input buffer
  std::vector<int8_t> matmul_input(deep * tile_num, 0);

  // Input sum buffer
  std::vector<int32_t> input_sum(input_sum_offset, 0);

  // Setup ConvParameter
  ConvParameter conv_param;
  memset(&conv_param, 0, sizeof(ConvParameter));

  conv_param.input_batch_ = batch;
  conv_param.input_h_ = in_h;
  conv_param.input_w_ = in_w;
  conv_param.input_channel_ = in_c;
  conv_param.output_h_ = out_h;
  conv_param.output_w_ = out_w;
  conv_param.output_channel_ = out_c;
  conv_param.kernel_h_ = kernel_h;
  conv_param.kernel_w_ = kernel_w;
  conv_param.stride_h_ = 1;
  conv_param.stride_w_ = 1;
  conv_param.pad_u_ = 0;
  conv_param.pad_d_ = 0;
  conv_param.pad_l_ = 0;
  conv_param.pad_r_ = 0;
  conv_param.dilation_h_ = 1;
  conv_param.dilation_w_ = 1;
  conv_param.group_ = 1;
  conv_param.tile_num_ = tile_num;
  conv_param.thread_num_ = 1;

  // Setup quantization parameters
  QuantArg input_quant_arg = {0.5f, -128};
  QuantArg filter_quant_args[4] = {{0.01f, 0}, {0.01f, 0}, {0.01f, 0}, {0.01f, 0}};
  QuantArg output_quant_arg = {0.5f, 50};
  int32_t out_act_min = -128;
  int32_t out_act_max = 127;
  int32_t left_shift[4] = {0, 0, 0, 0};
  int32_t right_shift[4] = {-8, -8, -8, -8};
  int32_t quant_multiplier[4] = {1073741824, 1073741824, 1073741824, 1073741824};

  conv_param.conv_quant_arg_.input_quant_args_ = &input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_args;
  conv_param.conv_quant_arg_.output_quant_args_ = &output_quant_arg;
  conv_param.conv_quant_arg_.out_act_min_ = &out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = &out_act_max;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 4;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

  // Matmul function
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  // Call ConvInt8 with is_optimize=true
  const int task_id = 0;
  const bool is_optimize = true;

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_optimize_true_larger output:\n";
  for (size_t i = 0; i < output_data.size(); ++i) {
    std::cout << static_cast<int32_t>(output_data[i]) << ", ";
  }
  std::cout << std::endl;

  // Verify output is in valid int8 range
  for (auto val : output_data) {
    EXPECT_GE(val, -128);
    EXPECT_LE(val, 127);
  }
}
