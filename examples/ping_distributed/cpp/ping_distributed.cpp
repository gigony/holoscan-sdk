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
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

class DummyOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyOp)

  DummyOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {}

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("Execution: {}", index_);
    sleep(1);
    index_++;
  }

  int index() const { return index_; }

 private:
  int index_ = 1;
};

class Fragment1 : public holoscan::Fragment {
 public:
  Fragment1(int64_t count = 10) : count_(count) {}

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<DummyOp>("tx", make_condition<CountCondition>(count_));
    add_operator(tx);
  }

 private:
  int64_t count_ = 10;
};

class Fragment2 : public holoscan::Fragment {
 public:
  Fragment2() = default;

  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<DummyOp>("rx", make_condition<CountCondition>(5));
    add_operator(rx);
  }
};

class App : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  void set_options(int64_t count = 10) { count_ = count; }

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<Fragment1>("fragment1", count_);
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    add_fragment(fragment1);
    add_fragment(fragment2);
  }

 private:
  int64_t count_ = 10;
};

std::optional<int64_t> get_int64_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (loc++ != std::end(args))) {
    try {
      return std::stoll(*loc);
    } catch (std::exception& e) {
      HOLOSCAN_LOG_ERROR("Unable to parse provided argument '{}'", name);
      return {};
    }
  }
  return {};
}

int main() {
  auto app = holoscan::make_application<App>();

  // Parse args that are defined for all applications.
  auto& remaining_args = app->argv();

  // Parse any additional supported arguments
  int64_t count = get_int64_arg(remaining_args, "--count").value_or(15);

  // configure tensor on host vs. GPU and set the count and shape
  app->set_options(count);

  // run the application
  app->run();

  return 0;
}
