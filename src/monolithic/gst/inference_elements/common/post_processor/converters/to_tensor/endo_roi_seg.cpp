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

#define NANONET_IMAGE_WIDTH  256
#define NANONET_IMAGE_HEIGHT 256


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


void EndoStreamer_CopyOutputBlobToGstStructure(InferenceBackend::OutputBlob::Ptr blob, GstStructure *gst_struct,
                                  const char *model_name, const char *layer_name, int32_t batch_size,
                                  int32_t batch_index) {
    try {
        if (!blob)
            throw std::invalid_argument("Blob pointer is null");

        const uint8_t *data = reinterpret_cast<const uint8_t *>(blob->GetData());
        if (data == nullptr)
            throw std::invalid_argument("Failed to get blob data");

        size_t size = GetUnbatchedSizeInBytes(blob, batch_size);

        // Data buffer pointer: const unit8_t *data; Data buffer length: size_t size 
            // Create a 1D float array from the raw data buffer
        std::vector<float> array1D(NANONET_IMAGE_WIDTH * NANONET_IMAGE_HEIGHT);
        memcpy(array1D.data(), data + batch_index * size, size);

        // Convert 1D array to 2D image
        cv::Mat _2d_image(NANONET_IMAGE_HEIGHT, NANONET_IMAGE_WIDTH, CV_32FC1, array1D.data());

        size_t input_width = getModelInputImageInfo().width;
        size_t input_height = getModelInputImageInfo().height;

        // Resize the image to exactly match To_Width and To_Height
        cv::Mat _new_2d_image_;
        cv::resize(_2d_image, _new_2d_image_, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

        // Convert the resized image back to a 1D float array
        cv::Mat _new_1d_ = _new_2d_image_.reshape(1, input_height * input_width); // Reshape to 1D


        // TODO: check data buffer size
        copy_buffer_to_structure(gst_struct, reinterpret_cast<float*>(_new_1d_.data), _new_1d_.total() * sizeof(float));

        gst_structure_set(gst_struct, "layer_name", G_TYPE_STRING, layer_name, "model_name", G_TYPE_STRING, model_name,
                          "precision", G_TYPE_INT, static_cast<int>(blob->GetPrecision()), "layout", G_TYPE_INT,
                          static_cast<int>(blob->GetLayout()), NULL);

    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("EndoStreamer_CopyOutputBlobToGstStructure Failed to copy model"));
    }
}


void EndoRoiSegConverter::processFrame_BuildSegmentationMaskTensor(const OutputBlob::Ptr &blob, const std::string &prefixed_layer_name, size_t batch_size, size_t frame_index, TensorsTable &tensors_table) const 
{

    size_t input_width = getModelInputImageInfo().width;
    size_t input_height = getModelInputImageInfo().height;


    GstStructure *tensor_data = BlobToTensorConverter::createTensor().gst_structure();

    GstStructure *tensor = tensor_data ; // For easy naming 

    EndoStreamer_CopyOutputBlobToGstStructure (blob, tensor, BlobToMetaConverter::getModelName().c_str(),
                                 prefixed_layer_name.c_str(), batch_size, frame_index);


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