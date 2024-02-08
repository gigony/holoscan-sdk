"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import ManualClock, RealtimeClock
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler


class MinimalOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        self.param_value = None
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def initialize(self):
        print("** initialize method called **")

    def start(self):
        print("** start method called **")

    def setup(self, spec: OperatorSpec):
        spec.param("param_value", 500)

    def stop(self):
        print("** stop method called **")

    def compute(self, op_input, op_output, context):
        self.count += 1


class MinimalApp(Application):
    def compose(self):
        mx = MinimalOp(self, CountCondition(self, 10), name="mx")
        self.add_operator(mx)


@pytest.mark.parametrize("SchedulerClass", [None, GreedyScheduler, MultiThreadScheduler])
def test_minimal_app(ping_config_file, SchedulerClass, capfd):  # noqa: N803
    app = MinimalApp()
    app.config(ping_config_file)
    if SchedulerClass is not None:
        app.scheduler(SchedulerClass(app))
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err

    # verify that the Python overrides of start and stop methods were called
    assert captured.out.count("** initialize method called **") == 1
    assert captured.out.count("** start method called **") == 1
    assert captured.out.count("** stop method called **") == 1


@pytest.mark.parametrize("SchedulerClass", [GreedyScheduler, MultiThreadScheduler])
@pytest.mark.parametrize("ClockClass", [RealtimeClock, ManualClock])
def test_minimal_app_with_clock(ping_config_file, SchedulerClass, ClockClass):  # noqa: N803
    app = MinimalApp()
    app.config(ping_config_file)
    app.scheduler(SchedulerClass(app, clock=ClockClass(app)))
    app.run()


def test_app_ping_config_keys(ping_config_file):
    app = MinimalApp()
    app.config(ping_config_file)
    keys = app.config_keys()
    assert isinstance(keys, set)
    assert keys == {"mx", "mx.multiplier"}


def test_app_config_keys(config_file):
    app = MinimalApp()
    app.config(config_file)
    keys = app.config_keys()
    assert isinstance(keys, set)

    # verify that various expected keys are present
    assert "aja.enable_overlay" in keys
    assert "replayer.basename" in keys
    assert "source" in keys
    assert "visualizer" in keys
    assert "visualizer.in_tensor_names" in keys

    # other non-existent keys are not
    assert "abcdefg" not in keys
