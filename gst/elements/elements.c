/*******************************************************************************
 * Copyright (C) <2018-2019> Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstgvaclassify.h"
#include "gstgvaidentify.h"
#include "gstgvainference.h"
#include "gstgvametaconvert.h"
#include "gstgvawatermark.h"
#include <gst/gst.h>

static gboolean plugin_init(GstPlugin *plugin) {
    if (!gst_element_register(plugin, "gvainference", GST_RANK_NONE, GST_TYPE_GVA_INFERENCE))
        return FALSE;

    if (!gst_element_register(plugin, "gvadetect", GST_RANK_NONE, GST_TYPE_GVA_INFERENCE))
        return FALSE;

    if (!gst_element_register(plugin, "gvaclassify", GST_RANK_NONE, GST_TYPE_GVA_CLASSIFY))
        return FALSE;

    if (!gst_element_register(plugin, "gvaidentify", GST_RANK_NONE, GST_TYPE_GVA_IDENTIFY))
        return FALSE;

    if (!gst_element_register(plugin, "gvametaconvert", GST_RANK_NONE, GST_TYPE_GVA_META_CONVERT))
        return FALSE;

    if (!gst_element_register(plugin, "gvawatermark", GST_RANK_NONE, GST_TYPE_GVA_WATERMARK))
        return FALSE;

    return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, videoanalytics, "Video Analytics elements", plugin_init,
                  PLUGIN_VERSION, PLUGIN_LICENSE, PACKAGE_NAME, GST_PACKAGE_ORIGIN)
