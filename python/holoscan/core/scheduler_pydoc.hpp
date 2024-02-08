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

#ifndef PYHOLOSCAN_CORE_SCHEDULER_PYDOC_HPP
#define PYHOLOSCAN_CORE_SCHEDULER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Scheduler {

PYDOC(Scheduler, R"doc(
Class representing a scheduler.
)doc")

//  Constructor
PYDOC(Scheduler_args_kwargs, R"doc(
Class representing a scheduler.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the Operator.

If a `fragment` keyword argument is provided, it must be of type
`holoscan.core.Fragment` (or
`holoscan.core.Application`). A single `Fragment` object can also be
provided positionally instead.

Any other arguments will be cast from a Python argument type to a C++ `Arg`
and stored in ``self.args``. (For details on how the casting is done, see the
`py_object_to_arg` utility).

Parameters
----------
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the scheduler.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the scheduler belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The scheduler's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the scheduler.
)doc")

PYDOC(initialize, R"doc(
initialization method for the scheduler.
)doc")

}  // namespace Scheduler

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_SCHEDULER_PYDOC_HPP
