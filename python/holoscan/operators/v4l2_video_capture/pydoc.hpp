/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
#define HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::V4L2VideoCaptureOp {

// V4L2VideoCaptureOp Constructor
PYDOC(V4L2VideoCaptureOp, R"doc(
Operator to get a video stream from a V4L2 source.

https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html

Inputs a video stream from a V4L2 node, including USB cameras and HDMI IN.

 - Input stream is on host. If no pixel format is specified in the yaml configuration file, the
   pixel format will be automatically selected. However, only ``AB24`` and ``YUYV`` are then
   supported.
   If a pixel format is specified in the yaml file, then this format will be used. However, note
   that the operator then expects that this format can be encoded as RGBA32. If not, the behavior
   is undefined.
 - Output stream is on host. Always RGBA32 at this time.

Use ``holoscan.operators.FormatConverterOp`` to move data from the host to a GPU device.

**==Named Outputs==**

    signal : nvidia::gxf::VideoBuffer
        A message containing a video buffer on the host with format GXF_VIDEO_FORMAT_RGBA.

Parameters
----------
fragment : Fragment (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
device : str
    The device to target (e.g. "/dev/video0" for device 0). Default value is ``"/dev/video0"``.
width : int, optional
    Width of the video stream. Default value is ``0``.
height : int, optional
    Height of the video stream. Default value is ``0``.
num_buffers : int, optional
    Number of V4L2 buffers to use. Default value is ``4``.
pixel_format : str
    Video stream pixel format (little endian four character code (fourcc)).
    Default value is ``"auto"``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"v4l2_video_capture"``.
exposure_time : int, optional
    Exposure time of the camera sensor in multiples of 100 μs (e.g. setting exposure_time to 100 is
    10 ms).
    Default: auto exposure, or camera sensor default.
    Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
    When not set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_AUTO, or to
    V4L2_EXPOSURE_APERTURE_PRIORITY if the former is not supported.
    When set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_SHUTTER_PRIORITY, or to
    V4L2_EXPOSURE_MANUAL if the former is not supported. The provided value is then used to set
    V4L2_CID_EXPOSURE_ABSOLUTE.
gain : int, optional
    Gain of the camera sensor.
    Default: auto gain, or camera sensor default.
    Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
    When not set by the user, V4L2_CID_AUTOGAIN is set to true (if supported).
    When set by the user, V4L2_CID_AUTOGAIN is set to false (if supported). The provided value is
    then used to set V4L2_CID_GAIN.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace holoscan::doc::V4L2VideoCaptureOp

#endif /* HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP */
