 // Testcase1: Basic test with small input size (1x5x5x8), stride=1, no padding
 TEST_F(ConvDw3x3Int8Test, ConvDw3x3Int8_Stride1_NoPad) {
   // Input: batch=1, h=5, w=5, channel=8
   // Output: batch=1, h=3, w=3, channel=8 (3x3 kernel, stride=1, valid padding)
   const int batch = 1;
   const int in_h = 5;
   const int in_w = 5;
   const int channel = 8;  // Must be multiple of 8 for ConvDw3x3Int8
   const int out_h = 3;
   const int out_w = 3;
   const int kernel_size = 3;
   const int stride = 1;

   // Simple input data pattern
   std::vector<int8_t> input = {
     // Channel 0
     -64, -32, 0, 32, 64,
     -32, -16, 0, 16, 32,
     0,   0,   0, 0,  0,
     32,  16,  0, -16, -32,
     64,  32,  0, -32, -64,
     // Channel 1
     32,  16,  0, -16, -32,
     16,  8,   0, -8,  -16,
     0,   0,   0, 0,   0,
     -16, -8,  0, 8,   16,
     -32, -16, 0, 16,  32,
     // Channel 2-7: fill with simple pattern
     1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
     61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
     91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
     105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
     120, 121, 122, 123, 124, 125, 126, 127
   };
   input.resize(batch * in_h * in_w * channel);

   // Weight: 3x3x8 kernel
   std::vector<int16_t> weight = {
     // Kernel for channel 0-7
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
   };

   // Bias
   std::vector<int32_t> bias(channel, 0);

   // Expected output (calculated manually for verification)
   std::vector<int8_t> benchmark = {
     // Output: 3x3x8
     -5, -2, 1, 4, 7, 10, 13, 16,
     -3, -1, 1, 3, 5, 7, 9, 11,
     -1, 0, 1, 2, 3, 4, 5, 6,
     1, 2, 3, 4, 5, 6, 7, 8,
     3, 4, 5, 6, 7, 8, 9, 10,
     5, 6, 7, 8, 9, 10, 11, 12,
     7, 8, 9, 10, 11, 12, 13, 14,
     9, 10, 11, 12, 13, 14, 15, 16,
     11, 12, 13, 14, 15, 16, 17, 18,
   };

   // Output buffer
   std::vector<int8_t> output(batch * out_h * out_w * channel, 0);

   // Buffer for intermediate computation (allocate on heap to avoid stack overflow)
   // Buffer size needs to be large enough for ConvDw3x3Int8InitBuffer
   // Based on code: block_input_h * block_input_w * 64, where block_input_h=3, block_input_w depends on stride
   int block_input_w = stride * (30 - 1) + 3;  // From code: block_output_w=30 for stride=1
   int buffer_size = 3 * block_input_w * 64;
   std::vector<int8_t> buffer(buffer_size, 0);

   // Setup ConvParameter
   ConvParameter conv_param;
   memset(&conv_param, 0, sizeof(ConvParameter));
   conv_param.kernel_h_ = kernel_size;
   conv_param.kernel_w_ = kernel_size;
   conv_param.stride_h_ = stride;
   conv_param.stride_w_ = stride;
   conv_param.pad_u_ = 0;
   conv_param.pad_d_ = 0;
   conv_param.pad_l_ = 0;
   conv_param.pad_r_ = 0;
   conv_param.dilation_h_ = 1;
   conv_param.dilation_w_ = 1;
   conv_param.input_batch_ = batch;
   conv_param.input_h_ = in_h;
   conv_param.input_w_ = in_w;
   conv_param.input_channel_ = channel;
   conv_param.output_batch_ = batch;
   conv_param.output_h_ = out_h;
   conv_param.output_w_ = out_w;
   conv_param.output_channel_ = channel;
   conv_param.thread_num_ = 1;
   conv_param.group_ = channel;

   // Setup quantization parameters
   conv_param.conv_quant_arg_.input_quant_args_ = new QuantArg[1];
   conv_param.conv_quant_arg_.input_quant_args_[0].scale_ = 1.0f;
   conv_param.conv_quant_arg_.input_quant_args_[0].zp_ = 0;
   conv_param.conv_quant_arg_.filter_quant_args_ = new QuantArg[1];
   conv_param.conv_quant_arg_.filter_quant_args_[0].scale_ = 1.0f;
   conv_param.conv_quant_arg_.filter_quant_args_[0].zp_ = 0;
   conv_param.conv_quant_arg_.output_quant_args_ = new QuantArg[1];
   conv_param.conv_quant_arg_.output_quant_args_[0].scale_ = 1.0f;
   conv_param.conv_quant_arg_.output_quant_args_[0].zp_ = 0;

   conv_param.conv_quant_arg_.quant_multiplier_ = new int32_t[channel];
   conv_param.conv_quant_arg_.left_shift_ = new int32_t[channel];
   conv_param.conv_quant_arg_.right_shift_ = new int32_t[channel];
   conv_param.conv_quant_arg_.out_act_min_ = new int32_t[1];
   conv_param.conv_quant_arg_.out_act_max_ = new int32_t[1];

   for (int i = 0; i < channel; i++) {
     conv_param.conv_quant_arg_.quant_multiplier_[i] = 1073741824;  // 1.0 in fixed point
     conv_param.conv_quant_arg_.left_shift_[i] = 0;
     conv_param.conv_quant_arg_.right_shift_[i] = 30;
   }
   conv_param.conv_quant_arg_.out_act_min_[0] = -128;
   conv_param.conv_quant_arg_.out_act_max_[0] = 127;
   conv_param.conv_quant_arg_.per_channel_ = 0;

   // Setup SlidingWindowParam
   SlidingWindowParam sliding;
   memset(&sliding, 0, sizeof(SlidingWindowParam));
   sliding.left_ = 0;
   sliding.right_ = out_w;
   sliding.top_ = 0;
   sliding.bottom_ = out_h;
   sliding.c_block_ = channel / 8;
   sliding.block_channel_ = channel;
   sliding.ic_align_ = channel;
   sliding.out_step_ = out_h * out_w * channel;
   sliding.out_h_step_ = out_w * channel;
   sliding.out_c_step_ = 1;
   sliding.out_w_step_ = channel;
   sliding.in_step_ = in_h * in_w * channel;
   sliding.in_h_step_ = in_w * channel;
   sliding.in_sh_step_ = in_w * channel * stride;
   sliding.in_sw_step_ = channel * stride;
   sliding.in_kh_step_ = in_w * channel;
   sliding.in_kw_step_ = channel;
   sliding.kernel_step_ = kernel_size * kernel_size * channel;

   // Run ConvDw3x3Int8
   int task_id = 0;
   ConvDw3x3Int8(output.data(), buffer.data(), input.data(), weight.data(), bias.data(),
                 &conv_param, &sliding, task_id);

   // Print output for debugging
   std::cout << "ConvDw3x3Int8Test-ConvDw3x3Int8_Stride1_NoPad output:\n";
   for (size_t i = 0; i < output.size(); i++) {
     std::cout << static_cast<int32_t>(output[i]) << ", ";
     if ((i + 1) % 8 == 0) std::cout << "\n";
   }
   std::cout << "\nConvDw3x3Int8Test-ConvDw3x3Int8_Stride1_NoPad benchmark:\n";
   for (size_t i = 0; i < benchmark.size(); i++) {
     std::cout << static_cast<int32_t>(benchmark[i]) << ", ";
     if ((i + 1) % 8 == 0) std::cout << "\n";
   }
   std::cout << std::endl;

   // Verify output size matches
   EXPECT_EQ(output.size(), benchmark.size());

   // Check cosine similarity
   float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
   std::cout << "Cosine similarity: " << similarity << std::endl;
   EXPECT_GT(similarity, 0.99f);

   // Cleanup
   delete[] conv_param.conv_quant_arg_.input_quant_args_;
   delete[] conv_param.conv_quant_arg_.filter_quant_args_;
   delete[] conv_param.conv_quant_arg_.output_quant_args_;
   delete[] conv_param.conv_quant_arg_.quant_multiplier_;
   delete[] conv_param.conv_quant_arg_.left_shift_;
   delete[] conv_param.conv_quant_arg_.right_shift_;
   delete[] conv_param.conv_quant_arg_.out_act_min_;
   delete[] conv_param.conv_quant_arg_.out_act_max_;
 }
