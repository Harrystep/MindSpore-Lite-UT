// Testcase1: Simple 1x1 convolution with is_optimize=false
// Input: 1x1x1x4, Output: 1x1x1x2
TEST_F(ConvInt8Test, ConvInt8_1x1_simple_no_opt) {
  // Use unique_ptr to allocate ConvParameter on heap
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 1;
  conv_param->input_w_ = 1;
  conv_param->input_channel_ = 4;
  conv_param->output_h_ = 1;
  conv_param->output_w_ = 1;
  conv_param->output_channel_ = 2;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 1;

  // Allocate quantization parameters on heap
  auto input_quant_arg = std::make_unique<QuantArg>();
  input_quant_arg->scale_ = 0.5f;
  input_quant_arg->zp_ = 0;

  auto filter_quant_arg = std::make_unique<QuantArg>();
  filter_quant_arg->scale_ = 0.25f;
  filter_quant_arg->zp_ = 0;

  auto output_quant_arg = std::make_unique<QuantArg>();
  output_quant_arg->scale_ = 1.0f;
  output_quant_arg->zp_ = 0;

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_arg.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 1;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = 0;

  // Allocate quantization arrays on heap
  auto quant_multiplier = std::make_unique<int32_t>(1073741824);
  auto left_shift = std::make_unique<int32_t>(0);
  auto right_shift = std::make_unique<int32_t>(0);
  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  // Data buffers on heap
  std::vector<int8_t> input_data = {-64, -32, 0, 32};

  int unit_size = 16;
  int up_round_oc = 4;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);

  // Weight: [[1, 2, 3, 4], [5, 6, 7, 8]]
  packed_weight[0] = 1;
  packed_weight[1] = 2;
  packed_weight[2] = 3;
  packed_weight[3] = 4;
  packed_weight[16] = 5;
  packed_weight[17] = 6;
  packed_weight[18] = 7;
  packed_weight[19] = 8;

  std::vector<int32_t> bias_data = {0, 0};
  std::vector<int32_t> filter_zp = {0, 0};
  std::vector<int8_t> output_data(2, 0);

  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 16 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, conv_param.get(), nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_1x1_simple_no_opt output: "
            << static_cast<int32_t>(output_data[0]) << " "
            << static_cast<int32_t>(output_data[1]) << std::endl;

  std::vector<int8_t> benchmark = {0, -128};
  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  ASSERT_GT(similarity, 0.99f);
}

// Testcase2: 1x1 convolution with 2x2 spatial input, is_optimize=false
TEST_F(ConvInt8Test, ConvInt8_1x1_2x2_no_opt) {
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 2;
  conv_param->input_w_ = 2;
  conv_param->input_channel_ = 2;
  conv_param->output_h_ = 2;
  conv_param->output_w_ = 2;
  conv_param->output_channel_ = 2;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 4;

  auto input_quant_arg = std::make_unique<QuantArg>(QuantArg{0.5f, 0});
  auto filter_quant_arg = std::make_unique<QuantArg>(QuantArg{0.25f, 0});
  auto output_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_arg.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 1;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = 0;

  auto quant_multiplier = std::make_unique<int32_t>(1073741824);
  auto left_shift = std::make_unique<int32_t>(0);
  auto right_shift = std::make_unique<int32_t>(0);
  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  std::vector<int8_t> input_data = {-64, 32, -32, 0, 16, -16, 48, -48};

  int unit_size = 16;
  int up_round_oc = 4;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
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
           0, conv_param.get(), nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_1x1_2x2_no_opt output: ";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  std::vector<int8_t> benchmark = {-96, -127, -32, -64, 32, 64, 96, 127};
  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  ASSERT_GT(similarity, 0.99f);
}

// Testcase3: is_optimize=true with simple 1x1 convolution
TEST_F(ConvInt8Test, ConvInt8_1x1_optimize) {
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 1;
  conv_param->input_w_ = 1;
  conv_param->input_channel_ = 4;
  conv_param->output_h_ = 1;
  conv_param->output_w_ = 1;
  conv_param->output_channel_ = 2;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 1;

  auto input_quant_arg = std::make_unique<QuantArg>(QuantArg{0.5f, 0});
  auto filter_quant_arg = std::make_unique<QuantArg>(QuantArg{0.25f, 0});
  auto output_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_arg.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 1;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = 0;

  auto quant_multiplier = std::make_unique<int32_t>(1073741824);
  auto left_shift = std::make_unique<int32_t>(0);
  auto right_shift = std::make_unique<int32_t>(0);
  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  std::vector<int8_t> input_data = {-64, -32, 0, 32};

  // For is_optimize=true: unit_size aligned to C4NUM, up_round_oc aligned to C8NUM
  int unit_size = 4;   // UP_ROUND(4, 4)
  int up_round_oc = 8;  // UP_ROUND(2, 8)
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  packed_weight[0] = 1;
  packed_weight[1] = 2;
  packed_weight[2] = 3;
  packed_weight[3] = 4;
  packed_weight[4] = 5;
  packed_weight[5] = 6;
  packed_weight[6] = 7;
  packed_weight[7] = 8;

  std::vector<int32_t> bias_data = {0, 0};
  std::vector<int32_t> filter_zp = {0, 0};
  std::vector<int8_t> output_data(2, 0);

  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, conv_param.get(), MatMulInt8_4x16_r, true);

  std::cout << "ConvInt8Test-ConvInt8_1x1_optimize output: "
            << static_cast<int32_t>(output_data[0]) << " "
            << static_cast<int32_t>(output_data[1]) << std::endl;

  std::vector<int8_t> benchmark = {0, -128};
  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  ASSERT_GT(similarity, 0.99f);
}

// Testcase4: is_optimize=true with per-channel quantization
TEST_F(ConvInt8Test, ConvInt8_optimize_per_channel) {
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 1;
  conv_param->input_w_ = 1;
  conv_param->input_channel_ = 2;
  conv_param->output_h_ = 1;
  conv_param->output_w_ = 1;
  conv_param->output_channel_ = 8;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 1;

  auto input_quant_arg = std::make_unique<QuantArg>(QuantArg{0.5f, 0});
  auto filter_quant_args = std::make_unique<QuantArg[]>(8);
  for (int i = 0; i < 8; i++) {
    filter_quant_args[i] = {0.25f, 0};
  }
  auto output_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_args.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 8;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

  auto quant_multiplier = std::make_unique<int32_t[]>(8);
  auto left_shift = std::make_unique<int32_t[]>(8);
  auto right_shift = std::make_unique<int32_t[]>(8);
  for (int i = 0; i < 8; i++) {
    quant_multiplier[i] = 1073741824;
    left_shift[i] = 0;
    right_shift[i] = 0;
  }

  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  std::vector<int8_t> input_data = {16, -16};

  int unit_size = 4;
  int up_round_oc = 8;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  for (int oc = 0; oc < 8; oc++) {
    for (int ic = 0; ic < 2; ic++) {
      packed_weight[oc * unit_size + ic] = (oc % 2 == 0) ? 1 : -1;
    }
  }

  std::vector<int32_t> bias_data(8, 0);
  std::vector<int32_t> filter_zp(8, 0);
  std::vector<int8_t> output_data(8, 0);

  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, conv_param.get(), MatMulInt8_4x16_r, true);

  std::cout << "ConvInt8Test-ConvInt8_optimize_per_channel output: ";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  bool has_nonzero = false;
  for (auto val : output_data) {
    if (val != 0) {
      has_nonzero = true;
      break;
    }
  }
  ASSERT_TRUE(has_nonzero);
}

// Testcase5: is_optimize=true with 2x2 spatial input
TEST_F(ConvInt8Test, ConvInt8_optimize_2x2) {
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 2;
  conv_param->input_w_ = 2;
  conv_param->input_channel_ = 2;
  conv_param->output_h_ = 2;
  conv_param->output_w_ = 2;
  conv_param->output_channel_ = 2;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 4;

  auto input_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});
  auto filter_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});
  auto output_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_arg.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 1;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = 0;

  auto quant_multiplier = std::make_unique<int32_t>(1073741824);
  auto left_shift = std::make_unique<int32_t>(0);
  auto right_shift = std::make_unique<int32_t>(0);
  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  std::vector<int8_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};

  int unit_size = 4;
  int up_round_oc = 8;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  packed_weight[0] = 1;
  packed_weight[1] = 0;
  packed_weight[4] = 0;
  packed_weight[5] = 1;

  std::vector<int32_t> bias_data = {0, 0};
  std::vector<int32_t> filter_zp = {0};
  std::vector<int8_t> output_data(8, 0);

  int kernel_plane = 1;
  int input_sum_size = 4 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 4, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 4 * 4, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, conv_param.get(), MatMulInt8_4x16_r, true);

  std::cout << "ConvInt8Test-ConvInt8_optimize_2x2 output: ";
  for (size_t i = 0; i < output_data.size(); i++) {
    std::cout << static_cast<int32_t>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  std::vector<int8_t> benchmark = {1, 2, 3, 4, 5, 6, 7, 8};
  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  ASSERT_GT(similarity, 0.99f);
}

// Testcase6: Minimal test with smallest possible dimensions
TEST_F(ConvInt8Test, ConvInt8_minimal) {
  auto conv_param = std::make_unique<ConvParameter>();
  memset(conv_param.get(), 0, sizeof(ConvParameter));
  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->pad_u_ = 0;
  conv_param->pad_d_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 1;
  conv_param->input_w_ = 1;
  conv_param->input_channel_ = 1;
  conv_param->output_h_ = 1;
  conv_param->output_w_ = 1;
  conv_param->output_channel_ = 1;
  conv_param->thread_num_ = 1;
  conv_param->tile_num_ = 1;

  auto input_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});
  auto filter_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});
  auto output_quant_arg = std::make_unique<QuantArg>(QuantArg{1.0f, 0});

  conv_param->conv_quant_arg_.input_quant_args_ = input_quant_arg.get();
  conv_param->conv_quant_arg_.filter_quant_args_ = filter_quant_arg.get();
  conv_param->conv_quant_arg_.output_quant_args_ = output_quant_arg.get();
  conv_param->conv_quant_arg_.input_arg_num_ = 1;
  conv_param->conv_quant_arg_.filter_arg_num_ = 1;
  conv_param->conv_quant_arg_.output_arg_num_ = 1;
  conv_param->conv_quant_arg_.per_channel_ = 0;

  auto quant_multiplier = std::make_unique<int32_t>(1073741824);
  auto left_shift = std::make_unique<int32_t>(0);
  auto right_shift = std::make_unique<int32_t>(0);
  auto act_min = std::make_unique<int32_t>(-128);
  auto act_max = std::make_unique<int32_t>(127);

  conv_param->conv_quant_arg_.quant_multiplier_ = quant_multiplier.get();
  conv_param->conv_quant_arg_.left_shift_ = left_shift.get();
  conv_param->conv_quant_arg_.right_shift_ = right_shift.get();
  conv_param->conv_quant_arg_.out_act_min_ = act_min.get();
  conv_param->conv_quant_arg_.out_act_max_ = act_max.get();

  std::vector<int8_t> input_data = {10};

  int unit_size = 16;
  int up_round_oc = 4;
  std::vector<int8_t> packed_weight(unit_size * up_round_oc, 0);
  packed_weight[0] = 2;  // weight = 2

  std::vector<int32_t> bias_data = {0};
  std::vector<int32_t> filter_zp = {0};
  std::vector<int8_t> output_data(1, 0);

  int kernel_plane = 1;
  int input_sum_size = 1 * up_round_oc;
  std::vector<int8_t> packed_input(unit_size * 1, 0);
  std::vector<int8_t> matmul_input(kernel_plane * 16 * 1, 0);
  std::vector<int32_t> input_sum(input_sum_size, 0);

  ConvInt8(input_data.data(), packed_input.data(), matmul_input.data(), packed_weight.data(),
           bias_data.data(), output_data.data(), filter_zp.data(), input_sum.data(),
           0, conv_param.get(), nullptr, false);

  std::cout << "ConvInt8Test-ConvInt8_minimal output: "
            << static_cast<int32_t>(output_data[0]) << std::endl;

  // Expected: 10 * 2 = 20
  std::vector<int8_t> benchmark = {20};
  float similarity = get_cosine_similarity_int8(output_data.data(), benchmark.data(), output_data.size());
  ASSERT_GT(similarity, 0.99f);
}
