/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/services/common/forward_op.hpp"

#include "holoscan/core/io_context.hpp"

namespace holoscan::ops {

void ForwardOp::setup(OperatorSpec& spec) {
  spec.input<std::any>("in");
  spec.output<std::any>("out");
}

void ForwardOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  auto in_message = op_input.receive<std::any>("in");
  if (in_message) { op_output.emit(in_message.value(), "out"); }
}

}  // namespace holoscan::ops
