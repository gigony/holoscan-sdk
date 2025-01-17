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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::HolovizOp {

// PyHolovizOp Constructor
PYDOC(HolovizOp, R"doc(
Holoviz visualization operator using Holoviz module.

This is a Vulkan-based visualizer.

**==Named Inputs==**

    receivers : multi-receiver accepting nvidia::gxf::Tensor and/or nvidia::gxf::VideoBuffer
        Any number of upstream ports may be connected to this ``receivers`` port. This port can
        accept either VideoBuffers or Tensors. These inputs can be in either host or device
        memory. Each tensor or video buffer will result in a layer. The operator autodetects the
        layer type for certain input types (e.g. a video buffer will result in an image layer). For
        other input types or more complex use cases, input specifications can be provided either at
        initialization time as a parameter or dynamically at run time (via ``input_specs``). On each
        call to ``compute``, tensors corresponding to all names specified in the ``tensors`` parameter
        must be found or an exception will be raised. Any extra, named tensors not present in the
        ``tensors`` parameter specification (or optional, dynamic ``input_specs`` input) will be
        ignored.
    input_specs : list[holoscan.operators.HolovizOp.InputSpec] (optional)
        A list of ``InputSpec`` objects. This port can be used to dynamically update the overlay
        specification at run time. No inputs are required on this port in order for the operator
        to ``compute``.
    render_buffer_input : nvidia::gxf::VideoBuffer (optional)
        An empty render buffer can optionally be provided. The video buffer must have format
        GXF_VIDEO_FORMAT_RGBA and be in device memory. This input port only exists if
        ``enable_render_buffer_input`` was set to ``True``, in which case ``compute`` will only be
        called when a message arrives on this input.

**==Named Outputs==**

    render_buffer_output : nvidia::gxf::VideoBuffer (optional)
        Output for a filled render buffer. If an input render buffer is specified, it is using
        that one, else it allocates a new buffer. The video buffer will have format
        GXF_VIDEO_FORMAT_RGBA and will be in device memory. This output is useful for offline
        rendering or headless mode. This output port only exists if ``enable_render_buffer_output``
        was set to ``True``.
    camera_pose_output : std::array<float, 16> (optional)
        The camera pose. The parameters returned represent the values of a 4x4 row major
        projection matrix. This output port only exists if ``enable_camera_pose_output`` was set to
        ``True``.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.core.Allocator, optional
    Allocator used to allocate render buffer output. If ``None``, will default to
    ``holoscan.core.UnboundedAllocator``.
receivers : sequence of holoscan.core.IOSpec, optional
    List of input receivers.
tensors : sequence of dict, optional
    List of input tensors. Each tensor is defined by a dictionary where the ``"name"`` key must
    correspond to a tensor sent to the operator's input. See the notes section below for further
    details on how the tensor dictionary is defined.
color_lut : list of list of float, optional
    Color lookup table for tensors of type ``color_lut``. Should be shape ``(n_colors, 4)``.
window_title : str, optional
    Title on window canvas. Default value is ``"Holoviz"``.
display_name : str, optional
    In exclusive mode, name of display to use as shown with xrandr. Default value is ``"DP-0"``.
width : int, optional
    Window width or display resolution width if in exclusive or fullscreen mode. Default value is
    ``1920``.
height : int, optional
    Window height or display resolution width if in exclusive or fullscreen mode. Default value is
    ``1080``.
framerate : float, optional
    Display framerate in Hz if in exclusive mode. Default value is ``60.0``.
use_exclusive_display : bool, optional
    Enable exclusive display. Default value is ``False``.
fullscreen : bool, optional
    Enable fullscreen window. Default value is ``False``.
headless : bool, optional
    Enable headless mode. No window is opened, the render buffer is output to
    port ``render_buffer_output``. Default value is ``False``.
enable_render_buffer_input : bool, optional
    If ``True``, an additional input port, named ``"render_buffer_input"`` is added to the
    operator. Default value is ``False``.
enable_render_buffer_output : bool, optional
    If ``True``, an additional output port, named ``"render_buffer_output"`` is added to the
    operator. Default value is ``False``.
enable_camera_pose_output : bool, optional.
    If ``True``, an additional output port, named ``"camera_pose_output"`` is added to the
    operator. Default value is ``False``.
font_path : str, optional
    File path for the font used for rendering text. Default value is ``""``.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams. Default value is
    ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"holoviz_op"``.

Notes
-----
The ``tensors`` argument is used to specify the tensors to display. Each tensor is defined using a
dictionary, that must, at minimum include a 'name' key that corresponds to a tensor found on the
operator's input. A 'type' key should also be provided to indicate the type of entry to display.
The 'type' key will be one of {``"color"``, ``"color_lut"``, ``"crosses"``, ``"lines"``,
``"lines_3d"``, ``"line_strip"``, ``"line_strip_3d"``, ``"ovals"``, ``"points"``, ``"points_3d"``,
``"rectangles"``, ``"text"``, ``"triangles"``, ``"triangles_3d"``, ``"depth_map"``,
``"depth_map_color"``, ``"unknown"``}. The default type is ``"unknown"`` which will attempt to
guess the corresponding type based on the tensor dimensions. Concrete examples are given below.

To show a single 2D RGB or RGBA image, use a list containing a single tensor of type ``"color"``.

.. code-block:: python

    tensors = [dict(name="video", type="color", opacity=1.0, priority=0)]

Here, the optional key ``opacity`` is used to scale the opacity of the tensor. The ``priority`` key
is used to specify the render priority for layers. Layers with a higher priority will be rendered
on top of those with a lower priority.

If we also had a ``"boxes"``` tensor representing rectangular bounding boxes, we could display them
on top of the image like this.

.. code-block:: python

    tensors = [
        dict(name="video", type="color", priority=0),
        dict(name="boxes", type="rectangles", color=[1.0, 0.0, 0.0], line_width=2, priority=1),
    ]

where the ``color`` and ``line_width`` keys specify the color and line width of the bounding box.

The details of the dictionary is as follows:

- **name**: name of the tensor containing the input data to display

  - type: ``str``
- **type**: input type (default ``"unknown"``)

  - type: ``str``
  - possible values:

    - **unknown**: unknown type, the operator tries to guess the type by inspecting the
      tensor.
    - **color**: RGB or RGBA color 2d image.
    - **color_lut**: single channel 2d image, color is looked up.
    - **points**: point primitives, one coordinate (x, y) per primitive.
    - **lines**: line primitives, two coordinates (x0, y0) and (x1, y1) per primitive.
    - **line_strip**: line strip primitive, a line primitive i is defined by each
      coordinate (xi, yi) and the following (xi+1, yi+1).
    - **triangles**: triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2)
      per primitive.
    - **crosses**: cross primitive, a cross is defined by the center coordinate and the
      size (xi, yi, si).
    - **rectangles**: axis aligned rectangle primitive, each rectangle is defined by two
      coordinates (xi, yi) and (xi+1, yi+1).
    - **ovals**: oval primitive, an oval primitive is defined by the center coordinate and
      the axis sizes (xi, yi, sxi, syi).
    - **text**: text is defined by the top left coordinate and the size (x, y, s) per
      string, text strings are defined by InputSpec member **text**.
    - **depth_map**: single channel 2d array where each element represents a depth value.
      The data is rendered as a 3d object using points, lines or triangles. The color for
      the elements can be specified through ``depth_map_color``. Supported format: 8-bit
      unsigned normalized format that has a single 8-bit depth component.
    - **depth_map_color**: RGBA 2d image, same size as the depth map. One color value for
      each element of the depth map grid. Supported format: 32-bit unsigned normalized
      format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, an
      8-bit B component in byte 2, and an 8-bit A component in byte 3.
- **opacity**: layer opacity, 1.0 is fully opaque, 0.0 is fully transparent (default:
  ``1.0``)

  - type: ``float``
- **priority**: layer priority, determines the render order, layers with higher priority
    values are rendered on top of layers with lower priority values (default: ``0``)

  - type: ``int``
- **color**: RGBA color of rendered geometry (default: ``[1.f, 1.f, 1.f, 1.f]``)

  - type: ``List[float]``
- **line_width**: line width for geometry made of lines (default: ``1.0``)

  - type: ``float``
- **point_size**: point size for geometry made of points (default: ``1.0``)

  - type: ``float``
- **text**: array of text strings, used when ``type`` is text (default: ``[]``)

  - type: ``List[str]``
- **depth_map_render_mode**: depth map render mode (default: ``points``)

  - type: ``str``
  - possible values:

    - **points**: render as points
    - **lines**: render as lines
    - **triangles**: render as triangles


1. Displaying Color Images

   Image data can either be on host or device (GPU). Multiple image formats are supported

   - R 8 bit unsigned
   - R 16 bit unsigned
   - R 16 bit float
   - R 32 bit unsigned
   - R 32 bit float
   - RGB 8 bit unsigned
   - BGR 8 bit unsigned
   - RGBA 8 bit unsigned
   - BGRA 8 bit unsigned
   - RGBA 16 bit unsigned
   - RGBA 16 bit float
   - RGBA 32 bit float

   When the ``type`` parameter is set to ``color_lut`` the final color is looked up using the values
   from the ``color_lut`` parameter. For color lookups these image formats are supported

   - R 8 bit unsigned
   - R 16 bit unsigned
   - R 32 bit unsigned

2. Drawing Geometry

   In all cases, ``x`` and ``y`` are normalized coordinates in the range ``[0, 1]``. The ``x`` and ``y``
   correspond to the horizontal and vertical axes of the display, respectively. The origin ``(0,
   0)`` is at the top left of the display.
   Geometric primitives outside of the visible area are clipped.
   Coordinate arrays are expected to have the shape ``(N, C)`` where ``N`` is the coordinate count
   and ``C`` is the component count for each coordinate.

   - Points are defined by a ``(x, y)`` coordinate pair.
   - Lines are defined by a set of two ``(x, y)`` coordinate pairs.
   - Lines strips are defined by a sequence of ``(x, y)`` coordinate pairs. The first two
     coordinates define the first line, each additional coordinate adds a line connecting to the
     previous coordinate.
   - Triangles are defined by a set of three ``(x, y)`` coordinate pairs.
   - Crosses are defined by ``(x, y, size)`` tuples. ``size`` specifies the size of the cross in the
     ``x`` direction and is optional, if omitted it's set to ``0.05``. The size in the ``y`` direction
     is calculated using the aspect ratio of the window to make the crosses square.
   - Rectangles (bounding boxes) are defined by a pair of 2-tuples defining the upper-left and
     lower-right coordinates of a box: ``(x1, y1), (x2, y2)``.
   - Ovals are defined by ``(x, y, size_x, size_y)`` tuples. ``size_x`` and ``size_y`` are optional, if
     omitted they are set to ``0.05``.
   - Texts are defined by ``(x, y, size)`` tuples. ``size`` specifies the size of the text in ``y``
     direction and is optional, if omitted it's set to ``0.05``. The size in the ``x`` direction is
     calculated using the aspect ratio of the window. The index of each coordinate references a
     text string from the ``text`` parameter and the index is clamped to the size of the text
     array. For example, if there is one item set for the ``text`` parameter, e.g.
     ``text=["my_text"]`` and three coordinates, then ``my_text`` is rendered three times. If
     ``text=["first text", "second text"]`` and three coordinates are specified, then ``first text``
     is rendered at the first coordinate, ``second text`` at the second coordinate and then ``second
     text`` again at the third coordinate. The ``text`` string array is fixed and can't be changed
     after initialization. To hide text which should not be displayed, specify coordinates
     greater than ``(1.0, 1.0)`` for the text item, the text is then clipped away.
   - 3D Points are defined by a ``(x, y, z)`` coordinate tuple.
   - 3D Lines are defined by a set of two ``(x, y, z)`` coordinate tuples.
   - 3D Lines strips are defined by a sequence of ``(x, y, z)`` coordinate tuples. The first two
     coordinates define the first line, each additional coordinate adds a line connecting to the
     previous coordinate.
   - 3D Triangles are defined by a set of three ``(x, y, z)`` coordinate tuples.

3. Displaying Depth Maps

   When ``type`` is ``depth_map`` the provided data is interpreted as a rectangular array of depth
   values. Additionally a 2d array with a color value for each point in the grid can be specified
   by setting ``type`` to ``depth_map_color``.

   The type of geometry drawn can be selected by setting ``depth_map_render_mode``.

   Depth maps are rendered in 3D and support camera movement. The camera is controlled using the
   mouse:

   - Orbit        (LMB)
   - Pan          (LMB + CTRL  | MMB)
   - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
   - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
   - Zoom         (Mouse wheel + SHIFT)

4. Output

   By default a window is opened to display the rendering, but the extension can also be run in
   headless mode with the ``headless`` parameter.

   Using a display in exclusive mode is also supported with the ``use_exclusive_display``
   parameter. This reduces the latency by avoiding the desktop compositor.

   The rendered framebuffer can be output to ``render_buffer_output``.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::HolovizOp

namespace holoscan::doc::HolovizOp::InputSpec {

// HolovizOp.InputSpec Constructor
PYDOC(InputSpec, R"doc(
InputSpec for the HolovizOp operator.

Parameters
----------
tensor_name : str
    The tensor name for this input.
type : holoscan.operators.HolovizOp.InputType or str
    The type of data that this tensor represents.

Attributes
----------
type : holoscan.operators.HolovizOp.InputType
    The type of data that this tensor represents.
opacity : float
    The opacity of the object. Must be in range [0.0, 1.0] where 1.0 is fully opaque.
priority : int
    Layer priority, determines the render order. Layers with higher priority values are rendered
    on top of layers with lower priority.
color : 4-tuple of float
    RGBA values in range [0.0, 1.0] for rendered geometry.
line_width : float
    Line width for geometry made of lines.
point_size : float
    Point size for geometry made of points.
text : sequence of str
    Sequence of strings used when type is `HolovizOp.InputType.TEXT`.
depth_map_render_mode : holoscan.operators.HolovizOp.DepthMapRenderMode
    The depth map render mode. Used only if `type` is `HolovizOp.InputType.DEPTH_MAP` or
    `HolovizOp.InputType.DEPTH_MAP_COLOR`.
views : list of HolovizOp.InputSpec.View
    Sequence of layer views. By default a layer will fill the whole window. When using a view, the
    layer can be placed freely within the window. When multiple views are specified, the layer is
    drawn multiple times using the specified layer views.
)doc")

PYDOC(InputSpec_description, R"doc(
Returns
-------
description : str
    YAML string representation of the InputSpec class.
)doc")

// HolovizOp.InputSpec.View Constructor
PYDOC(View, R"doc(
View for the InputSpec of a HolovizOp operator.

Attributes
----------
offset_x, offset_y : float
    Offset of top-left corner of the view. (0, 0) is the upper left and (1, 1) is the lower
    right.
width : float
    Normalized width (range [0.0, 1.0]).
height : float
    Normalized height (range [0.0, 1.0]).
matrix : sequence of float
    16-elements representing a 4x4 transformation matrix.

Notes
-----
Layers can also be placed in 3D space by specifying a 3D transformation `matrix`. Note that for
geometry layers there is a default matrix which allows coordinates in the range of [0 ... 1]
instead of the Vulkan [-1 ... 1] range. When specifying a matrix for a geometry layer, this
default matrix is overwritten.

When multiple views are specified, the layer is drawn multiple times using the specified layer
views.

It's possible to specify a negative term for height, which flips the image. When using a
negative height, one should also adjust the y value to point to the lower left corner of the
viewport instead of the upper left corner.
)doc")

}  // namespace holoscan::doc::HolovizOp::InputSpec

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_PYDOC_HPP */
