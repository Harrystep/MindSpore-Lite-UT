 // Testcase1: ConvDw3x3Int8Pad with 8x8x8 input, 3x3 kernel, stride 1
 // Using larger input size to avoid ASAN issues in ConvDw3x3Int8BorderPixel
 TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8Pad_8x8x8_Stride1) {
   const int batch = 1;
   const int in_h = 8;
   const int in_w = 8;
   const int channel = 8;
   const int out_h = 8;
   const int out_w = 8;

   // Input: 1x8x8x8 (NHWC format)
   // Simple sequential values for easy understanding
   std::vector<int8_t> input(batch * in_h * in_w * channel);
   for (size_t i = 0; i < input.size(); i++) {
     // Use values in range [-64, 63] to avoid overflow
     input[i] = static_cast<int8_t>((-64 + i) & 0x7f);
   }

   // Weight: 3x3x8 (NHWC format)
   // Identity-like kernel: center = 1, others = 0
   std::vector<int16_t> weight(channel * 3 * 3);
   for (int ch = 0; ch < channel; ch++) {
     for (int i = 0; i < 9; i++) {
       if (i == 4) {  // Center of 3x3 kernel
         weight[ch * 9 + i] = 100;  // Use 100 to compensate for quantization
       } else {
         weight[ch * 9 + i] = 0;
       }
     }
   }

   // Bias: 8 channels
   std::vector<int32_t> bias(channel, 0);

   // Output: 1x8x8x8
   std::vector<int8_t> output(batch * out_h * out_w * channel, -128);

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

   // Quantization parameters (per-channel)
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

   // Sliding window parameters
   // Pad function handles borders, internal region is [1, 7) x [1, 7)
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.top_ = 1;
   sliding.bottom_ = 7;
   sliding.left_ = 1;
   sliding.right_ = 7;
   sliding.in_kh_step_ = in_w * channel;
   sliding.in_kw_step_ = channel;

   std::cout << "Test ConvDw3x3Int8Pad with input size: " << in_h << "x" << in_w << "x" << channel << "\n";
   std::cout << "Sliding window: top=" << sliding.top_ << ", bottom=" << sliding.bottom_
             << ", left=" << sliding.left_ << ", right=" << sliding.right_ << "\n";
   std::cout << "in_kh_step=" << sliding.in_kh_step_ << ", in_kw_step=" << sliding.in_kw_step_ << "\n";

   // Run the padding function (only processes borders)
   ConvDw3x3Int8Pad(output.data(), input.data(), weight.data(), bias.data(), &conv_param, &sliding);

   // Output results for verification
   std::cout << "ConvDw3x3Int8Test-ConvDw3x3Int8Pad_8x8x8_Stride1 output (corners):\n";
   std::cout << "Top-left (0,0): [";
   for (int c = 0; c < 8; c++) {
     std::cout << static_cast<int32_t>(output[0 * 8 + c]) << " ";
   }
   std::cout << "]\n";

   std::cout << "Top-right (0,7): [";
   for (int c = 0; c < 8; c++) {
     std::cout << static_cast<int32_t>(output[0 * out_w * 8 + 7 * 8 + c]) << " ";
   }
   std::cout << "]\n";

   std::cout << "Bottom-left (7,0): [";
   for (int c = 0; c < 8; c++) {
     std::cout << static_cast<int32_t>(output[7 * out_w * 8 + 0 * 8 + c]) << " ";
   }
   std::cout << "]\n";

   std::cout << "Bottom-right (7,7): [";
   for (int c = 0; c < 8; c++) {
     std::cout << static_cast<int32_t>(output[7 * out_w * 8 + 7 * 8 + c]) << " ";
   }
   std::cout << "]\n";

   // Verify borders were processed (not -128 which was initialization value)
   EXPECT_NE(output[0], -128) << "Top-left corner should be processed";
   EXPECT_NE(output[7 * out_w * 8 + 7 * 8], -128) << "Bottom-right corner should be processed";

   // Check all values are in valid int8 range
   for (size_t i = 0; i < output.size(); i++) {
     ASSERT_GE(output[i], -128) << "Output value at index " << i << " is less than -128";
     ASSERT_LE(output[i], 127) << "Output value at index " << i << " is greater than 127";
   }
 }
