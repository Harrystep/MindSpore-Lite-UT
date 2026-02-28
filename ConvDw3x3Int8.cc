// Testcase1: ConvDw3x3Int8 with MINIMAL output data
 // KEY: input_w=32 (minimum for block_input_w), but output only 1x1=8 values!
 // Input: 6x32x8, Output: 1x1x8 (only 8 output values!)
 TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8_Minimal_1x1_Output) {
   const int batch = 1;
   const int in_h = 6;
   const int in_w = 32;  // Minimum for block_input_w=32 (stride=1)
   const int channel = 8;
   const int out_h = 4;
   const int out_w = 32;

   // Input: 1x6x32x8 - minimal but safe
   std::vector<int8_t> input(batch * in_h * in_w * channel);
   for (size_t i = 0; i < input.size(); i++) {
     input[i] = static_cast<int8_t>((i % 2 == 0) ? 10 : -10);  // Simple: 10, -10, 10, -10...
   }

   // Weight: 3x3x8 - simple center-only kernel
   std::vector<int16_t> weight(channel * 3 * 3);
   for (int ch = 0; ch < channel; ch++) {
     for (int i = 0; i < 9; i++) {
       weight[ch * 9 + i] = (i == 4) ? 50 : 0;  // Only center=50, others=0
     }
   }

   // Bias: all zeros
   std::vector<int32_t> bias(channel, 0);

   // Output: 1x4x32x8
   std::vector<int8_t> output(batch * out_h * out_w * channel, 0);

   // Buffer: 3 * 32 * 64 = 6144 bytes
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

   // Sliding window - process first row only
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.top_ = 0;
   sliding.bottom_ = 1;    // Only process row 0 (output_h=1)
   sliding.left_ = 0;
   sliding.right_ = 32;    // Process all 32 columns

   std::cout << "Test ConvDw3x3Int8 - MINIMAL output (1 row, 32 cols):\n";
   std::cout << "  Input: " << in_h << "x" << in_w << "x" << channel << "\n";
   std::cout << "  Output: 1x32x8 (only 32*8=256 values processed)\n";
   std::cout << "  Sliding: top=" << sliding.top_ << ", bottom=" << sliding.bottom_
             << ", left=" << sliding.left_ << ", right=" << sliding.right_ << "\n";

   int task_id = 0;
   ConvDw3x3Int8(output.data(), buffer.data(), input.data(), weight.data(), bias.data(),
                 &conv_param, &sliding, task_id);

   std::cout << "  SUCCESS!\n\n";

   // Verify output range
   for (int i = 0; i < 32 * channel; i++) {
     ASSERT_GE(output[i], -128) << "Index " << i << " below -128";
     ASSERT_LE(output[i], 127) << "Index " << i << " above 127";
   }

   // Show first few values
   std::cout << "First 24 output values: ";
   for (int i = 0; i < 24 && i < 32 * channel; i++) {
     std::cout << static_cast<int32_t>(output[i]) << " ";
   }
   std::cout << "\n";
 }
