/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "./serialization_buffers_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/serialization_buffer.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

class PySerializationBuffer : public SerializationBuffer {
 public:
  /* Inherit the constructors */
  using SerializationBuffer::SerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PySerializationBuffer(Fragment* fragment,
                                 std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                 size_t buffer_size = kDefaultSerializationBufferSize,
                                 const std::string& name = "serialization_buffer")
      : SerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyUcxSerializationBuffer : public UcxSerializationBuffer {
 public:
  /* Inherit the constructors */
  using UcxSerializationBuffer::UcxSerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxSerializationBuffer(Fragment* fragment,
                                    std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                    size_t buffer_size = kDefaultSerializationBufferSize,
                                    const std::string& name = "serialization_buffer")
      : UcxSerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_serialization_buffers(py::module_& m) {
  py::class_<SerializationBuffer,
             PySerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<SerializationBuffer>>(
      m, "SerializationBuffer", doc::SerializationBuffer::doc_SerializationBuffer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, size_t, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = "serialization_buffer"s,
           doc::SerializationBuffer::doc_SerializationBuffer_python)
      .def_property_readonly("gxf_typename",
                             &SerializationBuffer::gxf_typename,
                             doc::SerializationBuffer::doc_gxf_typename)
      .def("setup", &SerializationBuffer::setup, "spec"_a, doc::SerializationBuffer::doc_setup);

  py::class_<UcxSerializationBuffer,
             PyUcxSerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<UcxSerializationBuffer>>(
      m, "UcxSerializationBuffer", doc::UcxSerializationBuffer::doc_UcxSerializationBuffer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, size_t, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = "serialization_buffer"s,
           doc::UcxSerializationBuffer::doc_UcxSerializationBuffer_python)
      .def_property_readonly("gxf_typename",
                             &UcxSerializationBuffer::gxf_typename,
                             doc::UcxSerializationBuffer::doc_gxf_typename)
      .def("setup",
           &UcxSerializationBuffer::setup,
           "spec"_a,
           doc::UcxSerializationBuffer::doc_setup);
}
}  // namespace holoscan
