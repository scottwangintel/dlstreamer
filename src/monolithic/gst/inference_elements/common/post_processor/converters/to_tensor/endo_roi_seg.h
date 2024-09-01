/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once
//#error "hit this place"

#include "blob_to_tensor_converter.h"
#include "inference_backend/image_inference.h"

#include <gst/gst.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceBackend;

namespace post_processing {

class EndoRoiSegConverter : public BlobToTensorConverter {
  public:
    EndoRoiSegConverter(BlobToMetaConverter::Initializer initializer) : BlobToTensorConverter(std::move(initializer)) {
    }

    TensorsTable convert(const OutputBlobs &output_blobs) const override;

    static std::string getName() {
        return "endo_roi_seg";
    }

  private:
    void processFrame_RawCopy(const OutputBlob::Ptr &blob, const std::string &prefixed_layer_name, size_t batch_size, size_t frame_index, TensorsTable &tensors_table) const;
    void processFrame_BuildSegmentationMaskTensor(const OutputBlob::Ptr &blob, const std::string &prefixed_layer_name, size_t batch_size, size_t frame_index, TensorsTable &tensors_table) const;

};

} // namespace post_processing