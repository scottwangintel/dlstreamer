/*******************************************************************************
 * Copyright (C) 2021-2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "endo_roi_seg.h"
#include "copy_blob_to_gststruct.h"
#include "inference_backend/image_inference.h"
#include "inference_backend/logger.h"
#include "safe_arithmetic.hpp"

#include <exception>
#include <string>
#include <vector>

using namespace post_processing;
using namespace InferenceBackend;



TensorsTable EndoRoiSegConverter::convert(const OutputBlobs &output_blobs) const {
    ITT_TASK(__FUNCTION__);
    TensorsTable tensors_table;
    try {
        const size_t batch_size = getModelInputImageInfo().batch_size;
        tensors_table.resize(batch_size);

        for (const auto &blob_iter : output_blobs) {
            OutputBlob::Ptr blob = blob_iter.second;
            if (not blob) {
                throw std::invalid_argument("Output blob is empty");
            }

            const std::string &layer_name = blob_iter.first;
            const std::string prefixed_layer_name = "Endo-ROI-Seg2" + layer_name; // Add "Endo-" prefix
            const std::string prefixed_layer_name2 = "Endo-ROI-Seg2_2" + layer_name; // Add "Endo-" prefix


            for (size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
                //processFrame_RawCopy(blob, prefixed_layer_name, batch_size, frame_index, tensors_table);
                processFrame_BuildSegmentationMaskTensor(blob, prefixed_layer_name2, batch_size, frame_index, tensors_table);
            }
        }
    } catch (const std::exception &e) {
        GVA_ERROR("An error occurred while processing output BLOBs: %s", e.what());
    }
    return tensors_table;
}

void EndoRoiSegConverter::processFrame_RawCopy(const OutputBlob::Ptr &blob, const std::string &prefixed_layer_name, size_t batch_size, size_t frame_index, TensorsTable &tensors_table) const 
{
    GstStructure *tensor_data = BlobToTensorConverter::createTensor().gst_structure();

    CopyOutputBlobToGstStructure(blob, tensor_data, BlobToMetaConverter::getModelName().c_str(),
                                 prefixed_layer_name.c_str(), batch_size, frame_index);

    // In different versions of GStreamer, tensors_batch are attached to the buffer in a different order.
    // Thus, we identify our meta using tensor_id.
    gst_structure_set(tensor_data, "tensor_id", G_TYPE_INT, safe_convert<int>(frame_index), NULL);

    std::vector<GstStructure *> tensors{tensor_data};
    tensors_table[frame_index].push_back(tensors);
}



void EndoRoiSegConverter::processFrame_BuildSegmentationMaskTensor(const OutputBlob::Ptr &blob, const std::string &prefixed_layer_name, size_t batch_size, size_t frame_index, TensorsTable &tensors_table) const 
{

    size_t input_width = getModelInputImageInfo().width;
    size_t input_height = getModelInputImageInfo().height;


    GstStructure *tensor_data = BlobToTensorConverter::createTensor().gst_structure();

    GstStructure *tensor = tensor_data ; // For easy naming 

    EndoStreamer_CopyOutputBlobToGstStructure (blob, tensor, BlobToMetaConverter::getModelName().c_str(),
                                 prefixed_layer_name.c_str(), batch_size, frame_index, input_width, input_height);


    gst_structure_set_name(tensor, "mask_endo");
    gst_structure_set(tensor, "precision", G_TYPE_INT, GVA_PRECISION_FP32, NULL);
    gst_structure_set(tensor, "format", G_TYPE_STRING, "segmentation_mask", NULL);

    GValueArray *data0 = g_value_array_new(2);
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_TYPE_UINT);

    size_t masks_width = input_width;
    size_t masks_height = input_height;
    g_value_set_uint(&gvalue, safe_convert<uint32_t>(masks_height));
    g_value_array_append(data0, &gvalue);
    g_value_set_uint(&gvalue, safe_convert<uint32_t>(masks_width));
    g_value_array_append(data0, &gvalue);
    gst_structure_set_array(tensor, "dims", data0);
    g_value_array_free(data0);


    std::vector<GstStructure *> tensors{tensor};
    tensors_table[frame_index].push_back(tensors);

}