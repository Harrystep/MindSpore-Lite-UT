 TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8_6x6x8_Minimal_Stride1) {
   const int batch = 1;
   const int in_h = 6;
   const int in_w = 6;
   const int channel = 8;
   const int out_h = 4;
   const int out_w = 4;

   // Input: 1x6x6x8 (NHWC format) - only 288 bytes
   // Very simple data: sequential values starting from -32
   std::vector<int8_t> input(batch * in_h * in_w * channel);
   for (size_t i = 0; i < input.size(); i++) {
     input[i] = static_cast<int8_t>(-32 + (i % 64));  // Simple pattern
   }

   // Weight: 3x3x8 (NHWC format) - identity-like kernel
   // Center pixel has weight 1, others 0
   std::vector<int16_t> weight(channel * 3 * 3);
   for (int ch = 0; ch < channel; ch++) {
     for (int i = 0; i < 9; i++) {
       weight[ch * 9 + i] = (i == 4) ? 100 : 0;  // Only center = 100
     }
   }

   // Bias: 8 channels - all zeros
   std::vector<int32_t> bias(channel, 0);

   // Output: 1x4x4x8 - only 128 bytes
   std::vector<int8_t> output(batch * out_h * out_w * channel, 0);

   // Buffer - small temporary buffer
   // With input_w=6 < 150 and channel=8 < 64, will skip block processing
   std::vector<int8_t> buffer(1024, 0);  // 1KB is enough

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
   int32_t quant_multiplier[8] = {
     1073741824, 1073741824, 1073741824, 1073741824,
     1073741824, 1073741824, 1073741824, 1073741824
   };
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
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.top_ = 0;
   sliding.bottom_ = out_h;
   sliding.left_ = 0;
   sliding.right_ = out_w;

   std::cout << "Test ConvDw3x3Int8 with MINIMAL data:\n";
   std::cout << "  Input: " << in_h << "x" << in_w << "x" << channel << " (" << input.size() << " bytes)\n";
   std::cout << "  Output: " << out_h << "x" << out_w << "x" << channel << " (" << output.size() << " bytes)\n";

   // Run the convolution
   int task_id = 0;
   ConvDw3x3Int8(output.data(), buffer.data(), input.data(), weight.data(), bias.data(),
                 &conv_param, &sliding, task_id);

   std::cout << "  Execution completed successfully!\n\n";

   // Verify all output values are in valid int8 range
   for (size_t i = 0; i < output.size(); i++) {
     ASSERT_GE(output[i], -128) << "Value at index " << i << " is below -128";
     ASSERT_LE(output[i], 127) << "Value at index " << i << " is above 127";
   }

   // Check that output is not all zeros (some computation happened)
   int non_zero_count = 0;
   for (size_t i = 0; i < output.size(); i++) {
     if (output[i] != 0) {
       non_zero_count++;
     }
   }
   EXPECT_GT(non_zero_count, 0) << "Expected some non-zero output values";

   // Print simple output for verification
   std::cout << "Sample output (first 2x2 block, first 4 channels):\n";
   for (int h = 0; h < 2; h++) {
     for (int w = 0; w < 2; w++) {
       std::cout << "  [";
       for (int c = 0; c < 4; c++) {
         std::cout << static_cast<int32_t>(output[h * out_w * channel + w * channel + c]);
         if (c < 3) std::cout << " ";
       }
       std::cout << "]";
     }
     std::cout << "\n";
   }
 }