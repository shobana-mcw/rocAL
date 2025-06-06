/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <vector>

#include "video_loader.h"
//
// VideoLoaderSharded Can be used to run load and decode in multiple shards, each shard by a single loader instance,
// It improves load and decode performance since each loader loads the sequences in parallel using an internal thread
//
#ifdef ROCAL_VIDEO
class VideoLoaderSharded : public LoaderModule {
   public:
    explicit VideoLoaderSharded(void* dev_resources);
    ~VideoLoaderSharded() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size = false) override;
    void shut_down() override;
    void set_output(Tensor* output_image) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    CropImageInfo get_crop_image_info() override { return _crop_img_info; }
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override{};
    std::vector<size_t> get_sequence_start_frame_number() override;
    std::vector<std::vector<float>> get_sequence_frame_timestamps() override;
    Timing timing() override;
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char*>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) override {}

   private:
    void increment_loader_idx();
    void* _dev_resources;
    bool _initialized = false;
    std::vector<std::shared_ptr<VideoLoader>> _loaders;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();
    size_t _prefetch_queue_depth;  // Used for circular buffer's internal buffer
    Tensor* _output_tensor;
    CropImageInfo _crop_img_info;
};
#endif
