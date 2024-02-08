/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYBIND11_CORE_CORE_HPP
#define PYBIND11_CORE_CORE_HPP

#include <pybind11/pybind11.h>

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

void init_component(py::module_&);
void init_condition(py::module_&);
void init_network_context(py::module_&);
void init_resource(py::module_&);
void init_scheduler(py::module_&);
void init_executor(py::module_&);
void init_fragment(py::module_&);
void init_application(py::module_&);
void init_data_flow_tracker(py::module_&);
void init_cli(py::module_&);

// TODO: remove this unused function
template <typename ObjT>
std::vector<std::string> get_names_from_map(ObjT& map_obj) {
  std::vector<std::string> names;
  names.reserve(map_obj.size());
  for (auto& i : map_obj) { names.push_back(i.first); }
  return names;
}

}  // namespace holoscan

#endif /* PYBIND11_CORE_CORE_HPP */
