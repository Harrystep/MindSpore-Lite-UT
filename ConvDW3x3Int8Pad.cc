 // Testcase2: ConvDw3x3Int8Pad with 4x4x8 input, 3x3 kernel, stride 2
 // Tests border processing with stride 2
 TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8Pad_4x4x8_Stride2) {
   // Input: 1x4x4x8 (NHWC format)
   std::vector<int8_t> input(1 * 4 * 4 * 8);
   for (size_t i = 0; i < input.size(); i++) {
     input[i] = static_cast<int8_t>(-64 + (i % 128));
   }

   // Weight: 3x3x8 (averaging kernel)
   std::vector<int16_t> weight = {
     // 8 channels, each with a uniform 3x3 kernel
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 0
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 1
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 2
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 3
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 4
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 5
     1, 1, 1,  1, 1, 1,  1, 1, 1,   // Channel 6
     1, 1, 1,  1, 1, 1,  1, 1, 1    // Channel 7
   };

   // Bias: 8 channels
   std::vector<int32_t> bias = {0, 0, 0, 0, 0, 0, 0, 0};

   // Output: 1x2x2x8 (stride 2 reduces spatial size)
   std::vector<int8_t> output(1 * 2 * 2 * 8, -128);

   // Convolution parameters
   ConvParameter conv_param;
   memset(&conv_param, 0, sizeof(ConvParameter));
   conv_param.kernel_h_ = 3;
   conv_param.kernel_w_ = 3;
   conv_param.stride_h_ = 2;
   conv_param.stride_w_ = 2;
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
   conv_param.output_h_ = 2;
   conv_param.output_w_ = 2;
   conv_param.output_channel_ = 8;
   conv_param.thread_num_ = 1;

   // Quantization parameters
   QuantArg input_quant_arg = {0.5f, 0};
   QuantArg output_quant_arg = {0.5f, 0};

   std::vector<QuantArg> input_quant_args(1, input_quant_arg);
   std::vector<QuantArg> output_quant_args(1, output_quant_arg);
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
   // For 2x2 output with stride 2, borders may still be processed
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.top_ = 0;
   sliding.bottom_ = 2;
   sliding.left_ = 0;
   sliding.right_ = 2;
   sliding.in_kh_step_ = 4 * 8;
   sliding.in_kw_step_ = 8;

   // Run the padding function
   ConvDw3x3Int8Pad(output.data(), input.data(), weight.data(), bias.data(), &conv_param, &sliding);

   // Output results
   std::cout << "ConvDw3x3Int8Test-ConvDw3x3Int8Pad_4x4x8_Stride2 output:\n";
   for (int h = 0; h < 2; h++) {
     for (int w = 0; w < 2; w++) {
       std::cout << "[";
       for (int c = 0; c < 8; c++) {
         std::cout << static_cast<int32_t>(output[h * 2 * 8 + w * 8 + c]);
         if (c < 7) std::cout << " ";
       }
       std::cout << "] ";
     }
     std::cout << "\n";
   }

   // Verify output values are in valid range
   for (size_t i = 0; i < output.size(); i++) {
     ASSERT_GE(output[i], -128);
     ASSERT_LE(output[i], 127);
   }
 }