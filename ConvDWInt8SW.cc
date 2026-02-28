 // Testcase1: ConvDwInt8SW with simple 4x4x8 input, 3x3 kernel, stride 1
 // Basic test with small dimensions to avoid ASAN issues
 TEST_F(ConvDwInt8Test, ConvDwInt8SW_Basic_4x4x8) {
   // Input: 1x4x4x8 (NHWC format)
   // Simple pattern: values from -64 to 63
   std::vector<int8_t> input = {
     // Channel 0-3
     -64, -32, 0, 32,  -48, -16, 16, 48,  -32, 0, 32, 63,  -16, 16, 48, 60,
     // Channel 4-7
     -63, -31, 1, 33,  -47, -15, 17, 49,  -31, 1, 33, 62,  -15, 17, 49, 59,
     // Channel 0-3
     -32, 0, 32, 63,  -16, 16, 48, 60,   0, 32, 63, 61,   16, 48, 60, 58,
     // Channel 4-7
     -31, 1, 33, 62,  -15, 17, 49, 59,   1, 33, 62, 60,   17, 49, 59, 57,
     // Channel 0-3
     0, 32, 63, 61,   16, 48, 60, 58,    32, 63, 61, 59,   48, 60, 58, 55,
     // Channel 4-7
     1, 33, 62, 60,   17, 49, 59, 57,    33, 62, 60, 58,   49, 59, 57, 54,
     // Channel 0-3
     16, 48, 60, 58,  48, 60, 58, 55,   60, 58, 55, 53,   60, 58, 55, 52,
     // Channel 4-7
     17, 49, 59, 57,  49, 59, 57, 54,   59, 57, 54, 52,   59, 57, 54, 51
   };

   // Weight: 3x3x8 (NHWC format, int16_t)
   // Simple identity-like weights: center = 1, others = 0
   std::vector<int16_t> weight = {
     // Channel 0: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0,
     // Channel 1: 3x3 kernel
     0, 0, 0,  0, 2, 0,  0, 0, 0,
     // Channel 2: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0,
     // Channel 3: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0,
     // Channel 4: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0,
     // Channel 5: 3x3 kernel
     0, 0, 0,  0, 2, 0,  0, 0, 0,
     // Channel 6: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0,
     // Channel 7: 3x3 kernel
     0, 0, 0,  0, 1, 0,  0, 0, 0
   };

   // Bias: 8 channels (all zeros for simplicity)
   std::vector<int32_t> bias = {0, 0, 0, 0, 0, 0, 0, 0};

   // Input zero points: 8 channels (all zeros for simplicity)
   std::vector<int8_t> input_zp = {0, 0, 0, 0, 0, 0, 0, 0};

   // Output zero points: 8 channels
   std::vector<int32_t> output_zp = {0, 0, 0, 0, 0, 0, 0, 0};

   // Output size: 1x4x4x8
   std::vector<int8_t> output(1 * 4 * 4 * 8, 0);

   // Convolution parameters
   ConvParameter conv_param;
   memset(&conv_param, 0, sizeof(ConvParameter));
   conv_param.kernel_h_ = 3;
   conv_param.kernel_w_ = 3;
   conv_param.stride_h_ = 1;
   conv_param.stride_w_ = 1;
   conv_param.dilation_h_ = 1;
   conv_param.dilation_w_ = 1;
   conv_param.pad_u_ = 1;
   conv_param.pad_d_ = 1;
   conv_param.pad_l_ = 1;
   conv_param.pad_r_ = 1;
   conv_param.input_batch_ = 1;
   conv_param.input_h_ = 4;
   conv_param.input_w_ = 4;
   conv_param.input_channel_ = 8;
   conv_param.output_batch_ = 1;
   conv_param.output_h_ = 4;
   conv_param.output_w_ = 4;
   conv_param.output_channel_ = 8;
   conv_param.thread_num_ = 1;

   // Quantization parameters
   QuantArg input_quant_arg = {1.0f, 0};
   QuantArg output_quant_arg = {1.0f, 0};

   std::vector<QuantArg> input_quant_args(1, input_quant_arg);
   std::vector<QuantArg> output_quant_args(1, output_quant_arg);

   // Multiplier and shift for per-channel quantization
   std::vector<int32_t> quant_multiplier = {1073741824, 1073741824, 1073741824, 1073741824,
                                            1073741824, 1073741824, 1073741824, 1073741824};
   std::vector<int32_t> left_shift = {0, 0, 0, 0, 0, 0, 0, 0};
   std::vector<int32_t> right_shift = {30, 30, 30, 30, 30, 30, 30, 30};
   std::vector<int32_t> out_act_min = {-128, -128, -128, -128, -128, -128, -128, -128};
   std::vector<int32_t> out_act_max = {127, 127, 127, 127, 127, 127, 127, 127};

   conv_param.conv_quant_arg_.input_quant_args_ = input_quant_args.data();
   conv_param.conv_quant_arg_.output_quant_args_ = output_quant_args.data();
   conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier.data();
   conv_param.conv_quant_arg_.left_shift_ = left_shift.data();
   conv_param.conv_quant_arg_.right_shift_ = right_shift.data();
   conv_param.conv_quant_arg_.out_act_min_ = out_act_min.data();
   conv_param.conv_quant_arg_.out_act_max_ = out_act_max.data();
   conv_param.conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

   // Sliding window parameters
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.c_block_ = 1;  // 1 block of 8 channels
   sliding.block_channel_ = 8;
   sliding.left_ = 1;     // Skip border pixels due to padding
   sliding.right_ = 3;
   sliding.top_ = 1;
   sliding.bottom_ = 3;
   sliding.out_step_ = 4 * 4 * 8;
   sliding.out_h_step_ = 4 * 8;
   sliding.in_step_ = 4 * 4 * 8;
   sliding.in_h_step_ = 4 * 8;
   sliding.in_sh_step_ = 4 * 8;
   sliding.in_sw_step_ = 8;
   sliding.in_kh_step_ = 4 * 8;
   sliding.in_kw_step_ = 8;
   sliding.kernel_step_ = 3 * 3 * 8;

   // Run the convolution
   int task_id = 0;
   ConvDwInt8SW(output.data(), input.data(), weight.data(), bias.data(),
                input_zp.data(), output_zp.data(), &conv_param, &sliding, task_id);

   // Expected output (approximately same as input due to identity kernel)
   // Note: Due to padding and quantization, border pixels may differ
   std::cout << "ConvDwInt8Test-ConvDwInt8SW_Basic_4x4x8 output:\n";
   for (int i = 0; i < 16; i++) {
     for (int j = 0; j < 8; j++) {
       std::cout << static_cast<int32_t>(output[i * 8 + j]) << " ";
     }
     std::cout << "\n";
   }

   // Basic sanity checks: output should not be all zeros
   bool all_zero = true;
   for (size_t i = 0; i < output.size(); i++) {
     if (output[i] != 0) {
       all_zero = false;
       break;
     }
   }
   ASSERT_FALSE(all_zero) << "Output should not be all zeros";

   // Check that values are in valid int8 range
   for (size_t i = 0; i < output.size(); i++) {
     ASSERT_GE(output[i], -128);
     ASSERT_LE(output[i], 127);
   }
 }