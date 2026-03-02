TEST_F(ConvInt8Test, ConvInt8_minimal_1x1_opt) {
  // Input: batch=1, h=1, w=1, c=1
  std::vector<int8_t> input = {10};

  // Expected output: 10 * 2 = 20
  std::vector<int8_t> benchmark = {20};

  int output_size = 1 * 1 * 1 * 1;
  std::vector<int8_t> output(output_size, 0);

  // ConvParameter setup
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
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  // Quantization parameters - per-layer mode
  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;  // 1 << 30
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  // Buffer sizes for is_optimize=true
  int kernel_plane = 1 * 1;  // = 1
  int deep = kernel_plane * 1;  // = 1
#ifdef ENABLE_ARM32
  int unit_size = 4;   // UP_ROUND(deep, 4)
  int up_round_oc = 2;  // UP_ROUND(1, 2)
#else
  int unit_size = 4;   // UP_ROUND(deep, 4) = 4
  int up_round_oc = 8;  // UP_ROUND(1, 8) = 8
#endif
  int tile_n = 1;
  int input_sum_size = tile_n * up_round_oc;

  // Allocate buffers on heap via vector
  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(1, 0);

  // Set weight: output = input * weight
  packed_weight[0] = 2;  // weight = 2, so 10 * 2 = 20

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_minimal_1x1_opt output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nConvInt8Test-ConvInt8_minimal_1x1_opt benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: 2x2 spatial input with 1x1 kernel (is_optimize=true)
// Input: 1x2x2x1, Output: 1x2x2x1
TEST_F(ConvInt8Test, ConvInt8_2x2_spatial_opt) {
  // Input: batch=1, h=2, w=2, c=1
  std::vector<int8_t> input = {1, 2, 3, 4};

  // Expected output: input * 2
  std::vector<int8_t> benchmark = {2, 4, 6, 8};

  int output_size = 1 * 2 * 2 * 1;
  std::vector<int8_t> output(output_size, 0);

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
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 2;
  conv_param.output_w_ = 2;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 4;

  // Quantization parameters
  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int kernel_plane = 1 * 1;
  int deep = kernel_plane * 1;
#ifdef ENABLE_ARM32
  int unit_size = 4;
  int up_round_oc = 2;
#else
  int unit_size = 4;
  int up_round_oc = 8;
#endif
  int tile_n = 4;
  int input_sum_size = tile_n * up_round_oc;

  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(1, 0);

  packed_weight[0] = 2;  // weight = 2

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_2x2_spatial_opt output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nConvInt8Test-ConvInt8_2x2_spatial_opt benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: With bias test (is_optimize=true)
// Input: 1x1x1x1, Weight: 2, Bias: 5 -> Output: 10*2+5 = 25
TEST_F(ConvInt8Test, ConvInt8_with_bias_opt) {
  std::vector<int8_t> input = {10};
  std::vector<int8_t> benchmark = {25};

  std::vector<int8_t> output(1, 0);

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
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int kernel_plane = 1;
  int deep = kernel_plane * 1;
#ifdef ENABLE_ARM32
  int unit_size = 4;
  int up_round_oc = 2;
#else
  int unit_size = 4;
  int up_round_oc = 8;
#endif
  int tile_n = 1;
  int input_sum_size = tile_n * up_round_oc;

  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(1, 5);  // bias = 5

  packed_weight[0] = 2;  // weight = 2

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_with_bias_opt output: " << static_cast<int32_t>(output[0]) << std::endl;
  std::cout << "ConvInt8Test-ConvInt8_with_bias_opt benchmark: " << static_cast<int32_t>(benchmark[0]) << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: Two output channels (is_optimize=true)
// Input: 1x1x1x1 -> Output: 1x1x1x2
TEST_F(ConvInt8Test, ConvInt8_two_output_channels_opt) {
  std::vector<int8_t> input = {10};
  std::vector<int8_t> benchmark = {10, 20};  // 10*1, 10*2

  std::vector<int8_t> output(2, 0);

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
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 2;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int kernel_plane = 1;
  int deep = kernel_plane * 1;
#ifdef ENABLE_ARM32
  int unit_size = 4;
  int up_round_oc = 2;  // UP_ROUND(2, 2) = 2
#else
  int unit_size = 4;
  int up_round_oc = 8;  // UP_ROUND(2, 8) = 8
#endif
  int tile_n = 1;
  int input_sum_size = tile_n * up_round_oc;

  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(2, 0);

  // Weight for channel 0 = 1, channel 1 = 2
  packed_weight[0] = 1;   // channel 0
  packed_weight[4] = 2;   // channel 1 (offset by unit_size)

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_two_output_channels_opt output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nConvInt8Test-ConvInt8_two_output_channels_opt benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase5: Activation clipping test (is_optimize=true)
// Input: 100, Weight: 2 -> Result 200 should be clipped to 127
TEST_F(ConvInt8Test, ConvInt8_activation_clip_opt) {
  std::vector<int8_t> input = {100};
  std::vector<int8_t> benchmark = {127};  // 100*2=200, clipped to 127

  std::vector<int8_t> output(1, 0);

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
  conv_param.input_channel_ = 1;
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int kernel_plane = 1;
  int deep = kernel_plane * 1;
#ifdef ENABLE_ARM32
  int unit_size = 4;
  int up_round_oc = 2;
#else
  int unit_size = 4;
  int up_round_oc = 8;
#endif
  int tile_n = 1;
  int input_sum_size = tile_n * up_round_oc;

  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(1, 0);

  packed_weight[0] = 2;  // weight = 2, 100*2 = 200 -> clipped to 127

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_activation_clip_opt output: " << static_cast<int32_t>(output[0]) << std::endl;
  std::cout << "ConvInt8Test-ConvInt8_activation_clip_opt benchmark: " << static_cast<int32_t>(benchmark[0]) << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase6: Multiple input channels (is_optimize=true)
// Input: 1x1x1x2 -> Output: 1x1x1x1
TEST_F(ConvInt8Test, ConvInt8_multi_input_channel_opt) {
  // Input: 2 channels
  std::vector<int8_t> input = {10, 20};

  // Expected: 10*1 + 20*2 = 50
  std::vector<int8_t> benchmark = {50};

  std::vector<int8_t> output(1, 0);

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
  conv_param.input_channel_ = 2;  // 2 input channels
  conv_param.output_h_ = 1;
  conv_param.output_w_ = 1;
  conv_param.output_channel_ = 1;
  conv_param.thread_num_ = 1;
  conv_param.tile_num_ = 1;

  QuantArg input_quant_arg[1];
  QuantArg filter_quant_arg[1];
  QuantArg output_quant_arg[1];
  int32_t left_shift[1];
  int32_t right_shift[1];
  int32_t quant_multiplier[1];
  int32_t out_act_min[1];
  int32_t out_act_max[1];

  input_quant_arg[0].scale_ = 1.0f;
  input_quant_arg[0].zp_ = 0;
  filter_quant_arg[0].scale_ = 1.0f;
  filter_quant_arg[0].zp_ = 0;
  output_quant_arg[0].scale_ = 1.0f;
  output_quant_arg[0].zp_ = 0;

  left_shift[0] = 0;
  right_shift[0] = 0;
  quant_multiplier[0] = 1073741824;
  out_act_min[0] = -128;
  out_act_max[0] = 127;

  conv_param.conv_quant_arg_.input_quant_args_ = input_quant_arg;
  conv_param.conv_quant_arg_.filter_quant_args_ = filter_quant_arg;
  conv_param.conv_quant_arg_.output_quant_args_ = output_quant_arg;
  conv_param.conv_quant_arg_.left_shift_ = left_shift;
  conv_param.conv_quant_arg_.right_shift_ = right_shift;
  conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
  conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
  conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
  conv_param.conv_quant_arg_.input_arg_num_ = 1;
  conv_param.conv_quant_arg_.filter_arg_num_ = 1;
  conv_param.conv_quant_arg_.output_arg_num_ = 1;
  conv_param.conv_quant_arg_.per_channel_ = 0;

  int kernel_plane = 1;
  int deep = kernel_plane * 2;  // = 2 (2 input channels)
#ifdef ENABLE_ARM32
  int unit_size = 4;   // UP_ROUND(2, 4) = 4
  int up_round_oc = 2;
#else
  int unit_size = 4;   // UP_ROUND(2, 4) = 4
  int up_round_oc = 8;
#endif
  int tile_n = 1;
  int input_sum_size = tile_n * up_round_oc;

  std::vector<int8_t> packed_input(unit_size * tile_n, 0);
  std::vector<int8_t> matmul_input(deep * tile_n, 0);
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  std::vector<int32_t> filter_zp(1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);
  std::vector<int32_t> bias_data(1, 0);

  // Weight: channel0 weight = 1, channel1 weight = 2
  packed_weight[0] = 1;  // input channel 0 weight
  packed_weight[1] = 2;  // input channel 1 weight

  int task_id = 0;
  bool is_optimize = true;
  MATMUL_OPT_R_FUNC matmul_func = MatMulInt8_4x16_r;

  ConvInt8(input.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output.data(), filter_zp.data(), input_sum.data(),
           task_id, &conv_param, matmul_func, is_optimize);

  std::cout << "ConvInt8Test-ConvInt8_multi_input_channel_opt output: " << static_cast<int32_t>(output[0]) << std::endl;
  std::cout << "ConvInt8Test-ConvInt8_multi_input_channel_opt benchmark: " << static_cast<int32_t>(benchmark[0]) << std::endl;

  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
