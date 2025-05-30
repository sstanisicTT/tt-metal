# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_results,
)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0, 2.5, 5.5, 9.9])
def test_bw_softshrink(input_shapes, lambd, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    tt_output_tensor_on_device = ttnn.softshrink_bw(grad_tensor, input_tensor, lambd=lambd)

    golden_function = ttnn.get_golden_function(ttnn.softshrink_bw)
    golden_tensor = golden_function(grad_data, in_data, lambd)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0, 2.5, 5.5, 9.9])
def test_bw_softshrink_bf8b(input_shapes, lambd, device):
    in_data, input_tensor = data_gen_with_range_dtype(input_shapes, -100, 100, device, True, False, ttnn.bfloat8_b)
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -20, 20, device, False, False, ttnn.bfloat8_b)

    tt_output_tensor_on_device = ttnn.softshrink_bw(grad_tensor, input_tensor, lambd=lambd)

    golden_function = ttnn.get_golden_function(ttnn.softshrink_bw)
    golden_tensor = golden_function(grad_data, in_data, lambd)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
def test_bw_softshrink_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    tt_output_tensor_on_device = ttnn.softshrink_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.softshrink_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
