TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8_6x34x8_Stride1) {
   const int batch = 1;
   const int in_h = 6;
   const int in_w = 34;  // Must be >= block_input_w (32) for safety
   const int channel = 8;
   const int out_h = 4;
   const int out_w = 32;  // block_output_w=30, remainder=2

   // Input: 1x6x34x8
   std::vector<int8_t> input(batch * in_h * in_w * channel);
   for (size_t i = 0; i < input.size(); i++) {
     input[i] = static_cast<int8_t>(-32 + (i % 64));
   }

   // Weight: 3x3x8 - identity kernel
   std::vector<int16_t> weight(channel * 3 * 3);
   for (int ch = 0; ch < channel; ch++) {
     for (int i = 0; i < 9; i++) {
       weight[ch * 9 + i] = (i == 4) ? 100 : 0;
     }
   }

   // Bias
   std::vector<int32_t> bias(channel, 0);

   // Output: 1x4x32x8
   std::vector<int8_t> output(batch * out_h * out_w * channel, 0);

   // Buffer: 3 * 32 * 64
   std::vector<int8_t> buffer(3 * 32 * 64, 0);

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
   conv_param.input_batch_ = batch;
   conv_param.input_h_ = in_h;
   conv_param.input_w_ = in_w;
   conv_param.input_channel_ = channel;
   conv_param.output_batch_ = batch;
   conv_param.output_h_ = out_h;
   conv_param.output_w_ = out_w;
   conv_param.output_channel_ = channel;
   conv_param.thread_num_ = 1;

   // Quantization
   QuantArg input_quant_args[1] = {{1.0f, 0}};
   QuantArg output_quant_args[1] = {{1.0f, 0}};
   int32_t quant_multiplier[8] = {1073741824, 1073741824, 1073741824, 1073741824,
                                   1073741824, 1073741824, 1073741824, 1073741824};
   int32_t left_shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
   int32_t right_shift[8] = {30, 30, 30, 30, 30, 30, 30, 30};
   int32_t out_act_min[8] = {-128, -128, -128, -128, -128, -128, -128, -128};
   int32_t out_act_max[8] = {127, 127, 127, 127, 127, 127, 127, 127};

   conv_param.conv_quant_arg_.input_quant_args_ = input_quant_args;
   conv_param.conv_quant_arg_.output_quant_args_ = output_quant_args;
   conv_param.conv_quant_arg_.quant_multiplier_ = quant_multiplier;
   conv_param.conv_quant_arg_.left_shift_ = left_shift;
   conv_param.conv_quant_arg_.right_shift_ = right_shift;
   conv_param.conv_quant_arg_.out_act_min_ = out_act_min;
   conv_param.conv_quant_arg_.out_act_max_ = out_act_max;
   conv_param.conv_quant_arg_.per_channel_ = FILTER_PER_CHANNEL;

   // Sliding window - CRITICAL: must match actual output size!
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.top_ = 0;
   sliding.bottom_ = out_h;  // = 4
   sliding.left_ = 0;
   sliding.right_ = out_w;   // = 32

   std::cout << "Test ConvDw3x3Int8 stride=1:\n";
   std::cout << "  Input: " << in_h << "x" << in_w << "x" << channel << "\n";
   std::cout << "  Output: " << out_h << "x" << out_w << "x" << channel << "\n";
   std::cout << "  Sliding: top=" << sliding.top_ << ", bottom=" << sliding.bottom_
             << ", left=" << sliding.left_ << ", right=" << sliding.right_ << "\n";

   int task_id = 0;
   ConvDw3x3Int8(output.data(), buffer.data(), input.data(), weight.data(), bias.data(),
                 &conv_param, &sliding, task_id);

   std::cout << "  PASSED!\n\n";

   for (size_t i = 0; i < output.size(); i++) {
     ASSERT_GE(output[i], -128);
     ASSERT_LE(output[i], 127);
   }

   int non_zero = 0;
   for (size_t i = 0; i < output.size(); i++) {
     if (output[i] != 0) non_zero++;
   }
   EXPECT_GT(non_zero, 0);
 }
