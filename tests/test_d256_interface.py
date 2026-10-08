"""CPU routing checks; CUDA numerical and stream checks run separately."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


class D256InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.native = types.ModuleType("flash_attn_turing_d256")
        self.native.forward = mock.Mock(side_effect=lambda q, k, v, scale, causal:
                                        (torch.zeros_like(q), None))
        self.modules = mock.patch.dict(sys.modules, {
            "flash_attn_turing": types.ModuleType("flash_attn_turing"),
            "flash_attn_turing_d256": self.native,
        })
        self.modules.start()
        spec = importlib.util.spec_from_file_location(
            "interface_under_test", Path(__file__).parents[1] / "flash_attention_interface.py")
        self.interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.interface)

    def tearDown(self):
        self.modules.stop()

    def inputs(self, dim=256, requires_grad=False):
        return tuple(torch.zeros((1, 2, 2, dim), dtype=torch.float16,
                                 requires_grad=requires_grad) for _ in range(3))

    def test_default_scale_and_exact_inputs_reach_d256(self):
        qkv = self.inputs()
        result = self.interface.flash_attn_func(*qkv, causal=True)
        args = self.native.forward.call_args.args
        self.assertTrue(all(actual is expected for actual, expected in zip(args[:3], qkv)))
        self.assertEqual(args[3:], (0.0625, True))
        self.assertEqual(result.shape, qkv[0].shape)

    def test_explicit_scale_and_causal_flag_are_preserved(self):
        self.interface.flash_attn_func(*self.inputs(), softmax_scale=0.125, causal=False)
        self.assertEqual(self.native.forward.call_args.args[3:], (0.125, False))
        # Native contract tests must reject noncausal; the wrapper cannot silently change it.

    def test_backward_request_fails_before_native_call(self):
        for index in range(3):
            qkv = list(self.inputs())
            qkv[index].requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, "backward is unsupported"):
                self.interface.flash_attn_func(*qkv, causal=True)
        self.native.forward.assert_not_called()

    def test_no_grad_allows_inference_from_grad_tensors(self):
        qkv = self.inputs(requires_grad=True)
        with torch.no_grad():
            result = self.interface.flash_attn_func(*qkv, causal=True)
        self.native.forward.assert_called_once()
        self.assertFalse(result.requires_grad)

    def test_lower_dimensions_keep_the_existing_autograd_route(self):
        for dim in (64, 96, 128):
            qkv = self.inputs(dim, requires_grad=True)
            sentinel = object()
            with mock.patch.object(self.interface.FlashAttnFunc, "apply", return_value=sentinel) as apply:
                result = self.interface.flash_attn_func(*qkv, softmax_scale=0.25, causal=True)
                self.assertIs(result, sentinel)
                args = apply.call_args.args
                self.assertTrue(all(actual is expected for actual, expected in zip(args[:3], qkv)))
                self.assertEqual(args[3:], (0.25, True, True))
        self.native.forward.assert_not_called()

    def test_native_errors_are_not_hidden_by_a_fallback(self):
        self.native.forward.side_effect = RuntimeError("unsupported native contract")
        with self.assertRaisesRegex(RuntimeError, "unsupported native contract"):
            self.interface.flash_attn_func(*self.inputs(), causal=True)


if __name__ == "__main__":
    unittest.main()
