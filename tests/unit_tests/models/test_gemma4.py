# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Unit tests for Gemma-4: component tests, HF parity, shared KV, k_eq_v,
# inference, mixed precision, MoE, and end-to-end training tests.
#
# Single-GPU:
#   python -m pytest tests/unit_tests/models/test_gemma4.py -v
#   python -m pytest tests/unit_tests/models/test_gemma4.py -v -k "DualRoPE"
#   python -m pytest tests/unit_tests/models/test_gemma4.py -v -k "Parity"
#
# Multi-GPU (TP tests require WORLD_SIZE>=2):
#   NVIDIA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m pytest tests/unit_tests/models/test_gemma4.py -v -k "TensorParallel"

import argparse
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gemma4_layer_specs import (
    Gemma4RotaryEmbedding,
    Gemma4SelfAttention,
    Gemma4TransformerLayer,
    Gemma4TransformerLayerSubmodules,
    get_gemma4_layer_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.test_utilities import Utils

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _gelu_pytorch_tanh(x):
    return torch.nn.functional.gelu(x, approximate='tanh')


def _make_gemma4_config(**overrides) -> TransformerConfig:
    """Small Gemma-4-like TransformerConfig for testing.

    Defaults: 6 layers, hidden=128, 4Q/2KV GQA, head_dim=32, ffn=512,
    SWA window=4, skip_freq=6, RMSNorm, no bias, GEGLU.
    """
    defaults = dict(
        num_layers=6,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        global_kv_channels=64,
        num_global_query_groups=2,
        ffn_hidden_size=512,
        normalization='RMSNorm',
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.gelu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        window_size=(3, 0),
        window_attn_skip_freq=6,
        use_cpu_initialization=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _init_model_parallel_or_skip():
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)


# ---------------------------------------------------------------------------
# Phase 3: Gemma4RotaryEmbedding
# ---------------------------------------------------------------------------


class TestGemma4DualRoPEEmbedding:
    """Gemma4RotaryEmbedding: dual-theta RoPE construction and forward."""

    def setup_method(self):
        _init_model_parallel_or_skip()
        self.config = _make_gemma4_config(
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
        )
        self.rope = Gemma4RotaryEmbedding(
            config=self.config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_construction(self):
        assert isinstance(self.rope, Gemma4RotaryEmbedding)
        assert isinstance(self.rope.rope_sliding, RotaryEmbedding)
        assert isinstance(self.rope.rope_full, RotaryEmbedding)

    def test_forward_returns_tuple(self):
        result = self.rope(32)
        assert isinstance(result, tuple) and len(result) == 2

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_sliding_full_dim_difference(self):
        """Sliding emb uses head_dim; full emb uses global_head_dim with proportional RoPE."""
        emb_sliding, emb_full = self.rope(32)
        assert emb_sliding.shape[-1] == self.config.kv_channels
        assert emb_full.shape[-1] == self.config.global_kv_channels

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_different_frequencies(self):
        """Sliding and full embeddings differ because theta differs."""
        emb_sliding, emb_full = self.rope(32)
        shared_dim = min(emb_sliding.shape[-1], emb_full.shape[-1])
        assert not torch.allclose(emb_sliding[..., :shared_dim], emb_full[..., :shared_dim])

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_get_rotary_seq_len_delegates_to_sliding(self):
        seq_len = self.rope.get_rotary_seq_len(
            inference_context=None,
            transformer=None,
            transformer_input=torch.zeros(16, 2, 128, device='cuda'),
            transformer_config=self.config,
        )
        assert seq_len == 16


# ---------------------------------------------------------------------------
# Phase 3: dual-RoPE layer selection
# ---------------------------------------------------------------------------


class TestGemma4DualRoPELayerSelection:
    """Gemma4TransformerLayer selects the right RoPE tensor per layer number."""

    def setup_method(self):
        _init_model_parallel_or_skip()
        self.config = _make_gemma4_config(
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
        )
        self.spec = get_gemma4_layer_spec(self.config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_number,expect_full", [
        (1, False), (5, False), (6, True), (12, True), (2, False),
    ])
    def test_layer_type_selection_logic(self, layer_number, expect_full):
        assert (
            not is_layer_window_attention(
                self.config.window_size, self.config.window_attn_skip_freq, layer_number
            )
        ) == expect_full

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_dual_rope_forward_sliding_layer(self):
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        emb_s = torch.randn(sq, 1, 1, 32, device='cuda')
        emb_f = torch.randn(sq, 1, 1, 64, device='cuda')
        with torch.no_grad():
            out, ctx = layer(hidden_states=hidden, attention_mask=None,
                             rotary_pos_emb=(emb_s, emb_f))
        assert out.shape == (sq, b, h)
        assert ctx is None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_dual_rope_forward_full_layer(self):
        layer = build_module(self.spec, config=self.config, layer_number=6).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        emb_s = torch.randn(sq, 1, 1, 32, device='cuda')
        emb_f = torch.randn(sq, 1, 1, 64, device='cuda')
        with torch.no_grad():
            out, ctx = layer(hidden_states=hidden, attention_mask=None,
                             rotary_pos_emb=(emb_s, emb_f))
        assert out.shape == (sq, b, h)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_dual_rope_outputs_differ_by_layer_type(self):
        """Sliding and full layers produce different outputs from identical input."""
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        emb_s = torch.randn(sq, 1, 1, 32, device='cuda')
        emb_f = torch.randn(sq, 1, 1, 64, device='cuda')
        layer_s = build_module(self.spec, config=self.config, layer_number=1).cuda()
        layer_f = build_module(self.spec, config=self.config, layer_number=6).cuda()
        with torch.no_grad():
            out_s, _ = layer_s(hidden_states=hidden.clone(), attention_mask=None,
                               rotary_pos_emb=(emb_s, emb_f))
            out_f, _ = layer_f(hidden_states=hidden.clone(), attention_mask=None,
                               rotary_pos_emb=(emb_s, emb_f))
        assert not torch.allclose(out_s, out_f, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 4 + B: Per-Layer Embedding (after MLP) + v_norm
# ---------------------------------------------------------------------------


class TestGemma4PLELayer:
    """Phase 4: PLE applied to hidden states after attention + MLP (HF reference match)."""

    def setup_method(self):
        _init_model_parallel_or_skip()
        # per_layer_embed_dim=32 enables PLE modules in Gemma4TransformerLayer
        self.config = _make_gemma4_config(per_layer_embed_dim=32)
        self.spec = get_gemma4_layer_spec(self.config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_ple_modules_created_when_enabled(self):
        """per_layer_embed_dim > 0 → gate/projection/norm/scalar modules must exist."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        assert layer.per_layer_input_gate is not None
        assert layer.per_layer_projection is not None
        assert layer.post_per_layer_input_norm is not None
        assert layer.layer_scalar is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_layer_scalar_initialised_to_one(self):
        """layer_scalar must start at 1.0."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        assert torch.allclose(layer.layer_scalar, torch.ones(1, device='cuda'))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_per_layer_input_changes_output(self):
        """per_layer_input changes the layer output (applied after attention + MLP)."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        ple = torch.randn(sq, b, 32, device='cuda')

        with torch.no_grad():
            out_no_ple, _ = layer(hidden_states=hidden.clone(), attention_mask=None)
            out_ple, _ = layer(hidden_states=hidden.clone(), attention_mask=None,
                               per_layer_input=ple)

        assert not torch.allclose(out_no_ple, out_ple, atol=1e-5), (
            "PLE injection must change the output"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_no_per_layer_input_is_noop(self):
        """per_layer_input=None → same output as omitting the kwarg."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')

        with torch.no_grad():
            out_a, _ = layer(hidden_states=hidden.clone(), attention_mask=None)
            out_b, _ = layer(hidden_states=hidden.clone(), attention_mask=None,
                             per_layer_input=None)

        assert torch.allclose(out_a, out_b)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_per_layer_input_repeatable(self):
        """Same per_layer_input → identical output (deterministic)."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        ple = torch.randn(sq, b, 32, device='cuda')

        with torch.no_grad():
            out1, _ = layer(hidden_states=hidden.clone(), attention_mask=None,
                            per_layer_input=ple.clone())
            out2, _ = layer(hidden_states=hidden.clone(), attention_mask=None,
                            per_layer_input=ple.clone())

        assert torch.allclose(out1, out2)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_ple_disabled_when_dim_zero(self):
        """With per_layer_embed_dim=0 (default), PLE modules must not exist."""
        cfg = _make_gemma4_config()  # no per_layer_embed_dim
        spec = get_gemma4_layer_spec(cfg)
        layer = build_module(spec, config=cfg, layer_number=1).cuda()
        assert layer.per_layer_input_gate is None
        assert torch.allclose(layer.layer_scalar, torch.ones(1, device='cuda'))


# ---------------------------------------------------------------------------
# Phase B: v_norm on value states
# ---------------------------------------------------------------------------


class TestGemma4VNorm:
    """Phase B: Gemma4SelfAttention applies RMSNorm (no scale) to value states."""

    def setup_method(self):
        _init_model_parallel_or_skip()
        self.config = _make_gemma4_config()
        self.spec = get_gemma4_layer_spec(self.config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_attention_type_is_gemma4(self):
        """Spec must use Gemma4SelfAttention (which carries v_norm)."""
        assert self.spec.submodules.self_attention.module is Gemma4SelfAttention

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_forward_does_not_crash(self):
        """Layer forward must complete without error (v_norm applied silently)."""
        layer = build_module(self.spec, config=self.config, layer_number=1).cuda()
        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        with torch.no_grad():
            out, _ = layer(hidden_states=hidden, attention_mask=None)
        assert out.shape == (sq, b, h)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_full_attention_uses_global_head_dim_and_scale_one(self):
        layer = build_module(self.spec, config=self.config, layer_number=6).cuda()
        assert layer.self_attention.hidden_size_per_attention_head == self.config.global_kv_channels
        assert layer.self_attention.core_attention.softmax_scale == 1.0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_v_norm_normalises_values(self):
        """Gemma4SelfAttention.get_query_key_value_tensors applies unit-variance norm to values."""
        from megatron.core.models.backends import LocalSpecProvider
        from megatron.core.models.gpt.gemma4_layer_specs import Gemma4RMSNorm as RMSNorm
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import ModuleSpec

        backend = LocalSpecProvider()
        attn_spec = ModuleSpec(
            module=Gemma4SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=RMSNorm,
                k_layernorm=RMSNorm,
            ),
        )
        attn = build_module(attn_spec, config=self.config, layer_number=1).cuda()

        sq, b, h = 8, 1, 128
        hidden = torch.randn(sq, b, h, device='cuda')
        with torch.no_grad():
            q, k, v = attn.get_query_key_value_tensors(hidden)

        # v_norm: each head's values should have approximately unit L2 norm per dim
        # shape: [sq, b, num_kv_heads, head_dim]
        norms = v.pow(2).mean(-1)  # [sq, b, num_kv_heads]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
            f"v_norm should produce unit-variance values; got mean norm {norms.mean():.4f}"
        )


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestGemma4ConfigFields:
    """New config fields have correct defaults and are settable."""

    def test_dual_rope_fields_default_none(self):
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=2,
            use_cpu_initialization=True,
        )
        assert cfg.sliding_window_rope_base is None
        assert cfg.full_attention_rope_base is None
        assert cfg.full_attention_rope_partial_factor == 1.0

    def test_ple_fields_default_zero(self):
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=2,
            use_cpu_initialization=True,
        )
        assert cfg.per_layer_embed_vocab_size == 0
        assert cfg.per_layer_embed_dim == 0

    def test_dual_rope_fields_set(self):
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=2,
            use_cpu_initialization=True,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
        )
        assert cfg.sliding_window_rope_base == 10000.0
        assert cfg.full_attention_rope_base == 1000000.0
        assert cfg.full_attention_rope_partial_factor == 0.25

    def test_ple_fields_set(self):
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=2,
            use_cpu_initialization=True,
            per_layer_embed_vocab_size=262144,
            per_layer_embed_dim=256,
        )
        assert cfg.per_layer_embed_vocab_size == 262144
        assert cfg.per_layer_embed_dim == 256

    def test_global_attention_fields_set(self):
        cfg = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            use_cpu_initialization=True,
            global_kv_channels=32,
            num_global_query_groups=1,
            scale_embeddings_by_hidden_size=True,
        )
        assert cfg.global_kv_channels == 32
        assert cfg.num_global_query_groups == 1
        assert cfg.scale_embeddings_by_hidden_size is True

    def test_parser_accepts_gemma4_args_without_conflict(self):
        parser = argparse.ArgumentParser()
        add_megatron_arguments(parser)
        args, _ = parser.parse_known_args(
            [
                '--sliding-window-rope-base', '10000',
                '--full-attention-rope-base', '1000000',
                '--full-attention-rope-partial-factor', '0.25',
                '--per-layer-embed-vocab-size', '262144',
                '--per-layer-embed-dim', '256',
                '--global-kv-channels', '512',
                '--num-global-query-groups', '4',
                '--scale-embeddings-by-hidden-size',
                '--geglu',
            ]
        )
        assert args.sliding_window_rope_base == 10000
        assert args.full_attention_rope_base == 1000000
        assert args.full_attention_rope_partial_factor == 0.25
        assert args.per_layer_embed_vocab_size == 262144
        assert args.per_layer_embed_dim == 256
        assert args.global_kv_channels == 512
        assert args.num_global_query_groups == 4
        assert args.scale_embeddings_by_hidden_size is True
        assert args.geglu is True

    def test_gemma4_layer_spec_symbol(self):
        from megatron.core.models.gpt.gemma4_layer_specs import gemma4_layer_spec

        assert gemma4_layer_spec.module is Gemma4TransformerLayer


# ---------------------------------------------------------------------------
# HF Transformers numerical parity
# ---------------------------------------------------------------------------


class TestGemma4HFNumericalParity:
    """Numerical checks against HuggingFace Gemma-4 components.

    These tests intentionally compare small isolated components first. That makes
    parity failures actionable before moving to full-model logits parity.
    """

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _hf_config(self, hidden_size_per_layer_input: int = 0):
        configuration_gemma4 = pytest.importorskip(
            "transformers.models.gemma4.configuration_gemma4"
        )
        config = configuration_gemma4.Gemma4TextConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            max_position_embeddings=32,
            final_logit_softcapping=None,
            tie_word_embeddings=False,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        return config

    def _megatron_config(self, per_layer_embed_dim: int = 0):
        return _make_gemma4_config(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            per_layer_embed_vocab_size=64 if per_layer_embed_dim else 0,
            per_layer_embed_dim=per_layer_embed_dim,
            scale_embeddings_by_hidden_size=True,
            use_cpu_initialization=True,
        )

    def _build_megatron_attention(self, config, layer_number: int):
        from megatron.core.models.backends import LocalSpecProvider
        from megatron.core.models.gpt.gemma4_layer_specs import Gemma4RMSNorm as RMSNorm
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import ModuleSpec

        backend = LocalSpecProvider()
        attn_spec = ModuleSpec(
            module=Gemma4SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=RMSNorm,
                k_layernorm=RMSNorm,
            ),
        )
        return build_module(attn_spec, config=config, layer_number=layer_number).cuda().eval()

    @staticmethod
    def _fuse_qkv_gqa(q_weight, k_weight, v_weight, num_attention_heads, num_kv_heads, head_dim):
        hidden_size = q_weight.shape[1]
        num_q_per_group = num_attention_heads // num_kv_heads
        q = q_weight.view(num_kv_heads, num_q_per_group, head_dim, hidden_size)
        k = k_weight.view(num_kv_heads, 1, head_dim, hidden_size)
        v = v_weight.view(num_kv_heads, 1, head_dim, hidden_size)
        return torch.cat([q, k, v], dim=1).reshape(-1, hidden_size).contiguous()

    def _copy_attention_weights(self, megatron_attention, hf_attention):
        head_dim = hf_attention.head_dim
        num_attention_heads = hf_attention.config.num_attention_heads
        num_kv_heads = hf_attention.config.num_key_value_heads
        fused_qkv = self._fuse_qkv_gqa(
            hf_attention.q_proj.weight,
            hf_attention.k_proj.weight,
            hf_attention.v_proj.weight,
            num_attention_heads,
            num_kv_heads,
            head_dim,
        )
        megatron_attention.linear_qkv.weight.data.copy_(fused_qkv)
        megatron_attention.q_layernorm.weight.data.copy_(hf_attention.q_norm.weight)
        megatron_attention.k_layernorm.weight.data.copy_(hf_attention.k_norm.weight)
        megatron_attention.linear_proj.weight.data.copy_(hf_attention.o_proj.weight)

    @staticmethod
    def _hf_additive_attention_mask(seq_len: int, is_sliding: bool):
        from megatron.core.transformer.utils import (
            get_default_causal_mask,
            get_sliding_window_causal_mask,
        )

        if is_sliding:
            mask = get_sliding_window_causal_mask(seq_len, seq_len, (3, 0))
        else:
            mask = get_default_causal_mask(seq_len)

        additive_mask = torch.zeros(1, 1, seq_len, seq_len, device="cuda")
        return additive_mask.masked_fill(mask.view(1, 1, seq_len, seq_len), -10000.0)

    @staticmethod
    def _assert_gradient_close(actual, expected, max_abs: float, min_cosine: float):
        diff = (actual - expected).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten().float(),
            expected.flatten().float(),
            dim=0,
        )
        assert diff.max() <= max_abs, (
            f"max gradient diff {diff.max().item():.6g} exceeds {max_abs:.6g}"
        )
        assert cosine >= min_cosine, (
            f"gradient cosine similarity {cosine.item():.8f} is below {min_cosine:.8f}"
        )

    @staticmethod
    def _input_ids(batch: int, seq: int, vocab_size: int, pattern: str = "arange"):
        if pattern == "repeat":
            ids = torch.full((batch, seq), 7, device="cuda", dtype=torch.long)
        elif pattern == "reverse":
            ids = torch.arange(batch * seq, device="cuda", dtype=torch.long).view(batch, seq)
            ids = torch.flip(ids, dims=[1])
        else:
            ids = torch.arange(batch * seq, device="cuda", dtype=torch.long).view(batch, seq)
        return ids.remainder(vocab_size)

    @staticmethod
    def _position_ids(batch: int, seq: int):
        return torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)

    def _loader_args(self, hf_config):
        return SimpleNamespace(
            num_attention_heads=hf_config.num_attention_heads,
            group_query_attention=hf_config.num_key_value_heads != hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            num_global_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            global_kv_channels=hf_config.global_head_dim,
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                1 if layer_type == "sliding_attention" else 0
                for layer_type in hf_config.layer_types
            ],
            untie_embeddings_and_output_weights=not hf_config.tie_word_embeddings,
        )

    def _build_megatron_layer(self, config, layer_number: int):
        return build_module(
            get_gemma4_layer_spec(config),
            config=config,
            layer_number=layer_number,
        ).cuda()

    def _copy_layer_with_loader(self, megatron_layer, hf_layer, hf_config, layer_idx: int):
        from tools.checkpoint.loader_gemma4_hf import _set_layer_state

        model = SimpleNamespace(
            decoder=SimpleNamespace(layers=[None] * hf_config.num_hidden_layers)
        )
        hf_model = SimpleNamespace(
            model=SimpleNamespace(layers=[None] * hf_config.num_hidden_layers)
        )
        model.decoder.layers[layer_idx] = megatron_layer
        hf_model.model.layers[layer_idx] = hf_layer
        _set_layer_state(self._loader_args(hf_config), model, hf_model, layer_idx)

    def _build_megatron_model(self, config, hf_config):
        from megatron.core.models.gpt.gpt_model import GPTModel

        return GPTModel(
            config=config,
            transformer_layer_spec=get_gemma4_layer_spec(config),
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda()

    def _copy_model_with_loader(self, megatron_model, hf_model):
        from tools.checkpoint.loader_gemma4_hf import (
            _set_layer_state,
            _set_postprocess_state,
            _set_preprocess_state,
        )

        loader_args = self._loader_args(hf_model.config)
        _set_preprocess_state(megatron_model, hf_model)
        for layer_idx in range(hf_model.config.num_hidden_layers):
            _set_layer_state(loader_args, megatron_model, hf_model, layer_idx)
        _set_postprocess_state(loader_args, megatron_model, hf_model)

    def _megatron_last_hidden(self, model, input_ids, position_ids):
        preproc_output = model._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=None,
            inference_context=None,
            packed_seq_params=None,
            padding_mask=None,
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
            per_layer_inputs,
        ) = preproc_output[:7]
        ple_kwargs = {}
        if per_layer_inputs is not None:
            ple_kwargs["per_layer_inputs"] = per_layer_inputs
        return model.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=None,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            **ple_kwargs,
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.parametrize("layer_idx,layer_number", [(0, 1), (1, 2)])
    def test_attention_forward_matches_hf_transformers(self, layer_idx, layer_number):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(4242)
        hf_attention = modeling_gemma4.Gemma4TextAttention(hf_config, layer_idx=layer_idx).cuda().eval()
        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_attention = self._build_megatron_attention(megatron_config, layer_number)
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )
        self._copy_attention_weights(megatron_attention, hf_attention)

        batch, seq = 2, 5
        hidden_bsh = torch.randn(batch, seq, hf_config.hidden_size, device="cuda")
        hidden_sbh = hidden_bsh.transpose(0, 1).contiguous()
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        hf_mask = self._hf_additive_attention_mask(seq, hf_attention.is_sliding)
        megatron_pos_emb = megatron_rope(seq)[layer_idx]

        with torch.no_grad():
            hf_position_embeddings = hf_rope(
                hidden_bsh,
                position_ids,
                layer_type=hf_attention.layer_type,
            )
            hf_output, _ = hf_attention(
                hidden_states=hidden_bsh,
                position_embeddings=hf_position_embeddings,
                attention_mask=hf_mask,
                shared_kv_states={},
            )
            megatron_output, megatron_bias = megatron_attention(
                hidden_states=hidden_sbh,
                attention_mask=None,
                rotary_pos_emb=(megatron_pos_emb, megatron_pos_emb),
            )

        assert megatron_bias is None
        megatron_output = megatron_output.transpose(0, 1).contiguous()
        torch.testing.assert_close(megatron_output, hf_output, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_per_layer_inputs_match_hf_transformers(self):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        from megatron.core.models.gpt.gpt_model import GPTModel

        hf_config = self._hf_config(hidden_size_per_layer_input=8)
        megatron_config = self._megatron_config(per_layer_embed_dim=8)

        torch.manual_seed(3030)
        hf_model = modeling_gemma4.Gemma4TextModel(hf_config).cuda().eval()
        megatron_model = GPTModel(
            config=megatron_config,
            transformer_layer_spec=get_gemma4_layer_spec(megatron_config),
            vocab_size=hf_config.vocab_size,
            max_sequence_length=16,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda().eval()

        with torch.no_grad():
            megatron_model.embedding.word_embeddings.weight.copy_(hf_model.embed_tokens.weight)
            megatron_model.per_layer_embedding.weight.copy_(hf_model.embed_tokens_per_layer.weight)
            megatron_model.per_layer_model_proj.weight.copy_(
                hf_model.per_layer_model_projection.weight
            )
            megatron_model.per_layer_proj_norm.weight.copy_(hf_model.per_layer_projection_norm.weight)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

        with torch.no_grad():
            hf_inputs_embeds = hf_model.embed_tokens(input_ids)
            hf_per_layer = hf_model.get_per_layer_inputs(input_ids, hf_inputs_embeds)
            hf_per_layer = hf_model.project_per_layer_inputs(hf_inputs_embeds, hf_per_layer)

            preproc_output = megatron_model._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                inference_context=None,
                packed_seq_params=None,
                padding_mask=None,
            )
            megatron_input = preproc_output[0].transpose(0, 1).contiguous()
            megatron_per_layer = preproc_output[6]

        torch.testing.assert_close(megatron_input, hf_inputs_embeds, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(megatron_per_layer, hf_per_layer, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.parametrize("per_layer_embed_dim", [0, 8])
    @pytest.mark.parametrize("layer_idx,layer_number", [(0, 1), (1, 2)])
    def test_decoder_layer_forward_matches_hf_transformers(
        self, layer_idx, layer_number, per_layer_embed_dim
    ):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config(hidden_size_per_layer_input=per_layer_embed_dim)
        megatron_config = self._megatron_config(per_layer_embed_dim=per_layer_embed_dim)

        torch.manual_seed(5150)
        hf_layer = modeling_gemma4.Gemma4TextDecoderLayer(hf_config, layer_idx=layer_idx).cuda().eval()
        megatron_layer = self._build_megatron_layer(megatron_config, layer_number).eval()
        self._copy_layer_with_loader(megatron_layer, hf_layer, hf_config, layer_idx)
        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 6
        hidden_bsh = torch.randn(batch, seq, hf_config.hidden_size, device="cuda")
        hidden_sbh = hidden_bsh.transpose(0, 1).contiguous()
        position_ids = self._position_ids(batch, seq)
        hf_mask = self._hf_additive_attention_mask(seq, hf_layer.self_attn.is_sliding)
        per_layer_input_bsh = (
            torch.randn(batch, seq, per_layer_embed_dim, device="cuda")
            if per_layer_embed_dim
            else None
        )
        per_layer_input_sbh = (
            per_layer_input_bsh.transpose(0, 1).contiguous()
            if per_layer_input_bsh is not None
            else None
        )

        with torch.no_grad():
            hf_position_embeddings = hf_rope(
                hidden_bsh,
                position_ids,
                layer_type=hf_layer.self_attn.layer_type,
            )
            hf_output = hf_layer(
                hidden_states=hidden_bsh,
                per_layer_input=per_layer_input_bsh,
                shared_kv_states={},
                position_embeddings=hf_position_embeddings,
                attention_mask=hf_mask,
                position_ids=position_ids,
            )
            megatron_output, context = megatron_layer(
                hidden_states=hidden_sbh,
                attention_mask=None,
                rotary_pos_emb=megatron_rope(seq),
                per_layer_input=per_layer_input_sbh,
            )

        assert context is None
        megatron_output = megatron_output.transpose(0, 1).contiguous()
        torch.testing.assert_close(megatron_output, hf_output, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_loader_mapping_copies_expected_tensor_layouts(self):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config(hidden_size_per_layer_input=8)
        megatron_config = self._megatron_config(per_layer_embed_dim=8)

        torch.manual_seed(6161)
        hf_model = modeling_gemma4.Gemma4ForCausalLM(hf_config).cuda().eval()
        megatron_model = self._build_megatron_model(megatron_config, hf_config).eval()
        self._copy_model_with_loader(megatron_model, hf_model)

        torch.testing.assert_close(
            megatron_model.embedding.word_embeddings.weight,
            hf_model.model.embed_tokens.weight,
        )
        torch.testing.assert_close(
            megatron_model.per_layer_embedding.weight,
            hf_model.model.embed_tokens_per_layer.weight,
        )
        torch.testing.assert_close(
            megatron_model.decoder.final_layernorm.weight,
            hf_model.model.norm.weight,
        )
        torch.testing.assert_close(megatron_model.output_layer.weight, hf_model.lm_head.weight)

        for layer_idx, (megatron_layer, hf_layer) in enumerate(
            zip(megatron_model.decoder.layers, hf_model.model.layers)
        ):
            head_dim = hf_layer.self_attn.head_dim
            expected_qkv = self._fuse_qkv_gqa(
                hf_layer.self_attn.q_proj.weight,
                hf_layer.self_attn.k_proj.weight,
                hf_layer.self_attn.v_proj.weight,
                hf_config.num_attention_heads,
                hf_config.num_key_value_heads,
                head_dim,
            )
            expected_fc1 = torch.cat(
                [hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight],
                dim=0,
            )

            torch.testing.assert_close(megatron_layer.self_attention.linear_qkv.weight, expected_qkv)
            torch.testing.assert_close(
                megatron_layer.self_attention.linear_proj.weight,
                hf_layer.self_attn.o_proj.weight,
            )
            torch.testing.assert_close(megatron_layer.mlp.linear_fc1.weight, expected_fc1)
            torch.testing.assert_close(
                megatron_layer.mlp.linear_fc2.weight,
                hf_layer.mlp.down_proj.weight,
            )
            torch.testing.assert_close(
                megatron_layer.per_layer_input_gate.weight,
                hf_layer.per_layer_input_gate.weight,
                msg=f"PLE gate mismatch in layer {layer_idx}",
            )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.parametrize("per_layer_embed_dim", [0, 8])
    @pytest.mark.parametrize(
        "batch,seq,pattern",
        [
            (1, 1, "arange"),
            (1, 5, "reverse"),
            (2, 9, "repeat"),
        ],
    )
    def test_tiny_model_hidden_and_logits_match_hf_transformers(
        self, batch, seq, pattern, per_layer_embed_dim
    ):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config(hidden_size_per_layer_input=per_layer_embed_dim)
        megatron_config = self._megatron_config(per_layer_embed_dim=per_layer_embed_dim)

        torch.manual_seed(7171)
        hf_model = modeling_gemma4.Gemma4ForCausalLM(hf_config).cuda().eval()
        megatron_model = self._build_megatron_model(megatron_config, hf_config).eval()
        self._copy_model_with_loader(megatron_model, hf_model)

        input_ids = self._input_ids(batch, seq, hf_config.vocab_size, pattern)
        position_ids = self._position_ids(batch, seq)

        with torch.no_grad():
            hf_hidden = hf_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
            ).last_hidden_state
            hf_logits = hf_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
                logits_to_keep=0,
            ).logits
            megatron_hidden = self._megatron_last_hidden(
                megatron_model,
                input_ids,
                position_ids,
            ).transpose(0, 1).contiguous()
            megatron_logits = megatron_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        assert megatron_hidden.shape == hf_hidden.shape
        assert megatron_logits.shape == hf_logits.shape
        assert torch.isfinite(megatron_hidden).all()
        assert torch.isfinite(megatron_logits).all()
        torch.testing.assert_close(megatron_hidden, hf_hidden, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(megatron_logits, hf_logits, atol=2e-5, rtol=2e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_loader_full_model_output_matches_hf(self):
        """Loader-copied full GPTModel hidden states and logits match HF."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config(hidden_size_per_layer_input=8)
        megatron_config = self._megatron_config(per_layer_embed_dim=8)

        torch.manual_seed(7272)
        hf_model = modeling_gemma4.Gemma4ForCausalLM(hf_config).cuda().eval()
        megatron_model = self._build_megatron_model(megatron_config, hf_config).eval()
        self._copy_model_with_loader(megatron_model, hf_model)

        batch, seq = 2, 7
        input_ids = self._input_ids(batch, seq, hf_config.vocab_size, "reverse")
        position_ids = self._position_ids(batch, seq)

        with torch.no_grad():
            hf_hidden = hf_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
            ).last_hidden_state
            hf_logits = hf_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
                logits_to_keep=0,
            ).logits
            megatron_hidden = self._megatron_last_hidden(
                megatron_model,
                input_ids,
                position_ids,
            ).transpose(0, 1).contiguous()
            megatron_logits = megatron_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        torch.testing.assert_close(megatron_hidden, hf_hidden, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(megatron_logits, hf_logits, atol=2e-5, rtol=2e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.parametrize("layer_idx,layer_number", [(0, 1), (1, 2)])
    def test_decoder_layer_backward_matches_hf_transformers(self, layer_idx, layer_number):
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        per_layer_embed_dim = 8
        hf_config = self._hf_config(hidden_size_per_layer_input=per_layer_embed_dim)
        megatron_config = self._megatron_config(per_layer_embed_dim=per_layer_embed_dim)

        torch.manual_seed(8181)
        hf_layer = modeling_gemma4.Gemma4TextDecoderLayer(hf_config, layer_idx=layer_idx).cuda()
        megatron_layer = self._build_megatron_layer(megatron_config, layer_number)
        self._copy_layer_with_loader(megatron_layer, hf_layer, hf_config, layer_idx)
        hf_layer.train()
        megatron_layer.train()

        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 6
        hidden_bsh = torch.randn(
            batch, seq, hf_config.hidden_size, device="cuda", requires_grad=True
        )
        hidden_sbh = hidden_bsh.detach().clone().transpose(0, 1).contiguous().requires_grad_()
        per_layer_input_bsh = torch.randn(
            batch, seq, per_layer_embed_dim, device="cuda", requires_grad=True
        )
        per_layer_input_sbh = (
            per_layer_input_bsh.detach().clone().transpose(0, 1).contiguous().requires_grad_()
        )
        position_ids = self._position_ids(batch, seq)
        hf_mask = self._hf_additive_attention_mask(seq, hf_layer.self_attn.is_sliding)
        upstream = torch.randn(batch, seq, hf_config.hidden_size, device="cuda")

        hf_position_embeddings = hf_rope(
            hidden_bsh,
            position_ids,
            layer_type=hf_layer.self_attn.layer_type,
        )
        hf_output = hf_layer(
            hidden_states=hidden_bsh,
            per_layer_input=per_layer_input_bsh,
            shared_kv_states={},
            position_embeddings=hf_position_embeddings,
            attention_mask=hf_mask,
            position_ids=position_ids,
        )
        megatron_output, _ = megatron_layer(
            hidden_states=hidden_sbh,
            attention_mask=None,
            rotary_pos_emb=megatron_rope(seq),
            per_layer_input=per_layer_input_sbh,
        )
        megatron_output = megatron_output.transpose(0, 1).contiguous()

        (hf_output * upstream).sum().backward()
        (megatron_output * upstream).sum().backward()

        self._assert_gradient_close(
            hidden_sbh.grad.transpose(0, 1),
            hidden_bsh.grad,
            max_abs=7e-3,
            min_cosine=0.99999,
        )
        self._assert_gradient_close(
            per_layer_input_sbh.grad.transpose(0, 1),
            per_layer_input_bsh.grad,
            max_abs=2e-4,
            min_cosine=0.99999,
        )

        expected_qkv_grad = self._fuse_qkv_gqa(
            hf_layer.self_attn.q_proj.weight.grad,
            hf_layer.self_attn.k_proj.weight.grad,
            hf_layer.self_attn.v_proj.weight.grad,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_layer.self_attn.head_dim,
        )
        expected_fc1_grad = torch.cat(
            [hf_layer.mlp.gate_proj.weight.grad, hf_layer.mlp.up_proj.weight.grad],
            dim=0,
        )
        self._assert_gradient_close(
            megatron_layer.self_attention.linear_qkv.weight.grad,
            expected_qkv_grad,
            max_abs=8e-2,
            min_cosine=0.999,
        )
        torch.testing.assert_close(
            megatron_layer.mlp.linear_fc1.weight.grad,
            expected_fc1_grad,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            megatron_layer.per_layer_input_gate.weight.grad,
            hf_layer.per_layer_input_gate.weight.grad,
            atol=1e-4,
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Step 3: Shared KV Cache parity tests
# ---------------------------------------------------------------------------


class TestGemma4SharedKVParity:
    """Tests for shared KV cache (num_kv_shared_layers).

    Setup: 3 layers — [sliding, full, sliding] with num_kv_shared_layers=1.
    Layer 0 (sliding): source — stores K/V after forward.
    Layer 1 (full):    normal — stores K/V (no shared full layers, but still set).
    Layer 2 (sliding): shared — borrows K/V from layer 0 via _kv_source.

    HF stores post-RoPE K/V in ``shared_kv_states``. Megatron stores the
    rotary-ready pre-RoPE K/V and applies the same RoPE in the shared layer,
    which is numerically equivalent for full-sequence training forwards.
    """

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(3001)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _hf_config(self):
        configuration_gemma4 = pytest.importorskip(
            "transformers.models.gemma4.configuration_gemma4"
        )
        cfg = configuration_gemma4.Gemma4TextConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention", "full_attention"],
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=0,
            max_position_embeddings=32,
            final_logit_softcapping=None,
            tie_word_embeddings=False,
            use_cache=False,
        )
        # num_kv_shared_layers may not be a constructor param; set directly
        cfg.num_kv_shared_layers = 1
        cfg._attn_implementation = "eager"
        return cfg

    def _megatron_config(self):
        cfg = _make_gemma4_config(
            num_layers=3,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=[1, 0, 0],  # sliding, full, full
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            use_cpu_initialization=True,
        )
        # Attach Gemma4-specific fields that TransformerConfig doesn't know about
        cfg.num_kv_shared_layers = 1
        return cfg

    def _build_layers(self, config, n=3):
        return [
            build_module(
                get_gemma4_layer_spec(config), config=config, layer_number=i + 1
            ).cuda()
            for i in range(n)
        ]

    def _wire(self, layers):
        import torch.nn as nn

        from megatron.core.models.gpt.gemma4_layer_specs import wire_gemma4_kv_sharing
        wire_gemma4_kv_sharing(nn.ModuleList(layers))

    # ------------------------------------------------------------------
    # Flag detection (structural, no HF dependency)
    # ------------------------------------------------------------------

    def test_layer_flags_detected_correctly(self):
        """is_kv_shared_layer / store_full_length_kv / kv_shared_layer_index."""
        config = self._megatron_config()
        layers = self._build_layers(config)
        attn = [l.self_attention for l in layers]

        # Layer 0 (sliding, non-shared): only sliding in [0,1], so store_full_length_kv=True
        assert not attn[0].is_kv_shared_layer
        assert attn[0].store_full_length_kv
        assert attn[0].kv_shared_layer_index is None

        # Layer 1 (full, non-shared): store_full_length_kv=True (only full in [0,1])
        assert not attn[1].is_kv_shared_layer
        assert attn[1].store_full_length_kv

        # Layer 2 (full, shared): borrows KV from layer 1 (last non-shared full layer)
        assert attn[2].is_kv_shared_layer
        assert not attn[2].store_full_length_kv
        assert attn[2].kv_shared_layer_index == 1

    def test_wire_sets_kv_source_reference(self):
        """wire_gemma4_kv_sharing must link _kv_source to the correct attention module."""
        config = self._megatron_config()
        layers = self._build_layers(config)
        attn = [l.self_attention for l in layers]

        # Before wiring: _kv_source is None
        assert attn[2]._kv_source is None

        self._wire(layers)

        # After wiring: layer 2 → layer 1 (last non-shared full-attention layer)
        assert attn[2]._kv_source is attn[1]
        # Non-shared layers need no source
        assert attn[0]._kv_source is None
        assert attn[1]._kv_source is None

    # ------------------------------------------------------------------
    # Weight loading (HF dependency)
    # ------------------------------------------------------------------

    def test_loader_zeros_kv_rows_for_shared_layer(self):
        """Shared layers have no k_proj/v_proj in HF; linear_qkv K/V rows must be zero."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        from tools.checkpoint.loader_gemma4_hf import _set_layer_state

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(3010)
        hf_layers = [
            modeling_gemma4.Gemma4TextDecoderLayer(hf_config, layer_idx=i).cuda()
            for i in range(3)
        ]
        megatron_layers = self._build_layers(megatron_config)

        import types
        model = types.SimpleNamespace(decoder=types.SimpleNamespace(layers=megatron_layers))
        hf_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=hf_layers))
        loader_args = types.SimpleNamespace(
            num_layers=3,
            num_attention_heads=hf_config.num_attention_heads,
            group_query_attention=True,
            num_query_groups=hf_config.num_key_value_heads,
            num_global_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            global_kv_channels=hf_config.global_head_dim,
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                1 if lt == "sliding_attention" else 0 for lt in hf_config.layer_types
            ],
            untie_embeddings_and_output_weights=True,
            num_kv_shared_layers=1,
            attention_k_eq_v=False,
            enable_moe_block=False,
        )

        for i in range(3):
            _set_layer_state(loader_args, model, hf_model, i)

        # Layer 2 is shared full-attention: K/V rows in linear_qkv must be zero.
        # Full-attention uses global_head_dim for head dimensions.
        qkv_w = megatron_layers[2].self_attention.linear_qkv.weight.data
        num_kv = hf_config.num_key_value_heads
        num_q_per_group = hf_config.num_attention_heads // num_kv
        head_dim_full = hf_config.global_head_dim  # layer 2 is full-attention

        group_size_full = (num_q_per_group + 2) * head_dim_full
        for g in range(num_kv):
            group_rows = qkv_w[g * group_size_full: (g + 1) * group_size_full]
            k_rows = group_rows[num_q_per_group * head_dim_full: (num_q_per_group + 1) * head_dim_full]
            v_rows = group_rows[(num_q_per_group + 1) * head_dim_full:]
            assert k_rows.abs().max() == 0, f"K rows not zero in group {g} of shared layer"
            assert v_rows.abs().max() == 0, f"V rows not zero in group {g} of shared layer"

        # Layer 1 (source, full-attention): K/V rows must be non-zero (real weights)
        head_dim_sliding = hf_config.head_dim
        group_size_full1 = (num_q_per_group + 2) * head_dim_full
        qkv_w1 = megatron_layers[1].self_attention.linear_qkv.weight.data
        group1 = qkv_w1[:group_size_full1]
        k1 = group1[num_q_per_group * head_dim_full: (num_q_per_group + 1) * head_dim_full]
        assert k1.abs().max() > 0, "K rows should be non-zero for source layer (layer 1)"

    # ------------------------------------------------------------------
    # Forward: structural sanity (finite output, correct shape)
    # ------------------------------------------------------------------

    def test_shared_layer_forward_produces_finite_output(self):
        """Full forward pass through 3 layers (including shared layer) stays finite."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(3020)
        megatron_layers = self._build_layers(megatron_config)
        self._wire(megatron_layers)

        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden = torch.randn(seq, batch, hf_config.hidden_size, device="cuda")
        rope_emb = megatron_rope(seq)

        with torch.no_grad():
            h = hidden
            for layer in megatron_layers:
                h, _ = layer(h, attention_mask=None, rotary_pos_emb=rope_emb)

        assert h.shape == hidden.shape, "Output shape mismatch"
        assert torch.isfinite(h).all(), "Shared-KV forward produced non-finite values"

    def test_source_layer_stored_kv_shapes(self):
        """Source layer must populate _stored_kv with correct shapes after forward."""
        megatron_config = self._megatron_config()
        layers = self._build_layers(megatron_config)
        self._wire(layers)

        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden = torch.randn(seq, batch, megatron_config.hidden_size, device="cuda")

        with torch.no_grad():
            layers[0](hidden, attention_mask=None, rotary_pos_emb=megatron_rope(seq))

        attn0 = layers[0].self_attention
        assert attn0._stored_kv is not None, "Source layer did not populate _stored_kv"
        stored_k, stored_v = attn0._stored_kv
        # Shape should be [s, b, nkv, head_dim] (Megatron seq-first, pre-transpose)
        assert stored_k is not None
        assert stored_v is not None
        assert torch.isfinite(stored_k).all()
        assert torch.isfinite(stored_v).all()

    def test_shared_kv_decoder_stack_matches_hf_transformers(self):
        """Three-layer stack with shared-KV produces the same hidden states as HF."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        from tools.checkpoint.loader_gemma4_hf import _set_layer_state

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(3030)
        hf_layers = [
            modeling_gemma4.Gemma4TextDecoderLayer(hf_config, layer_idx=i).cuda().eval()
            for i in range(3)
        ]
        megatron_layers = [layer.eval() for layer in self._build_layers(megatron_config)]
        self._wire(megatron_layers)

        model = SimpleNamespace(decoder=SimpleNamespace(layers=megatron_layers))
        hf_model = SimpleNamespace(model=SimpleNamespace(layers=hf_layers))
        loader_args = SimpleNamespace(
            num_layers=3,
            num_attention_heads=hf_config.num_attention_heads,
            group_query_attention=True,
            num_query_groups=hf_config.num_key_value_heads,
            num_global_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            global_kv_channels=hf_config.global_head_dim,
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                1 if lt == "sliding_attention" else 0 for lt in hf_config.layer_types
            ],
            untie_embeddings_and_output_weights=True,
            num_kv_shared_layers=1,
            attention_k_eq_v=False,
            enable_moe_block=False,
        )
        for i in range(3):
            _set_layer_state(loader_args, model, hf_model, i)

        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden_bsh = torch.randn(batch, seq, hf_config.hidden_size, device="cuda")
        hidden_sbh = hidden_bsh.transpose(0, 1).contiguous()
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        shared_kv_states = {}

        with torch.no_grad():
            hf_hidden = hidden_bsh
            for hf_layer in hf_layers:
                hf_position_embeddings = hf_rope(
                    hf_hidden,
                    position_ids,
                    layer_type=hf_layer.self_attn.layer_type,
                )
                hf_mask = TestGemma4HFNumericalParity._hf_additive_attention_mask(
                    seq,
                    hf_layer.self_attn.is_sliding,
                )
                hf_hidden = hf_layer(
                    hidden_states=hf_hidden,
                    per_layer_input=None,
                    shared_kv_states=shared_kv_states,
                    position_embeddings=hf_position_embeddings,
                    attention_mask=hf_mask,
                    position_ids=position_ids,
                )

            megatron_hidden = hidden_sbh
            rope_emb = megatron_rope(seq)
            for megatron_layer in megatron_layers:
                megatron_hidden, _ = megatron_layer(
                    megatron_hidden,
                    attention_mask=None,
                    rotary_pos_emb=rope_emb,
                )
            megatron_hidden = megatron_hidden.transpose(0, 1).contiguous()

        torch.testing.assert_close(megatron_hidden, hf_hidden, atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------------------
# Step 4: attention_k_eq_v parity tests
# ---------------------------------------------------------------------------


class TestGemma4KEqualsVParity:
    """Tests for attention_k_eq_v (full-attention layers use K projection for V).

    Config: 2 layers [sliding, full] with attention_k_eq_v=True.
    - Sliding layer: k_eq_v flag is False (only applies to full-attention).
    - Full-attention layer: k_eq_v flag is True; value = v_norm(raw k_proj(x)).
    """

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(4001)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _hf_config(self):
        configuration_gemma4 = pytest.importorskip(
            "transformers.models.gemma4.configuration_gemma4"
        )
        cfg = configuration_gemma4.Gemma4TextConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
            num_global_key_value_heads=2,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=0,
            max_position_embeddings=32,
            final_logit_softcapping=None,
            tie_word_embeddings=False,
            use_cache=False,
            attention_k_eq_v=True,
        )
        cfg._attn_implementation = "eager"
        return cfg

    def _megatron_config(self):
        cfg = _make_gemma4_config(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,  # every 2nd layer is full-attention
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            use_cpu_initialization=True,
        )
        cfg.attention_k_eq_v = True
        return cfg

    def _build_attn(self, config, layer_number):
        from megatron.core.models.backends import LocalSpecProvider
        from megatron.core.models.gpt.gemma4_layer_specs import Gemma4RMSNorm as RMSNorm
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import ModuleSpec

        backend = LocalSpecProvider()
        spec = ModuleSpec(
            module=Gemma4SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=RMSNorm,
                k_layernorm=RMSNorm,
            ),
        )
        return build_module(spec, config=config, layer_number=layer_number).cuda().eval()

    # ------------------------------------------------------------------
    # Flag detection (no HF)
    # ------------------------------------------------------------------

    def test_k_eq_v_flag_only_on_full_attention_layers(self):
        """attention_k_eq_v must be True for full-attention, False for sliding."""
        config = self._megatron_config()
        # layer_number=1 → sliding, layer_number=2 → full (window_attn_skip_freq=2)
        sliding_attn = self._build_attn(config, layer_number=1)
        full_attn = self._build_attn(config, layer_number=2)

        assert not sliding_attn.attention_k_eq_v, "Sliding layer must NOT have k_eq_v"
        assert full_attn.attention_k_eq_v, "Full-attention layer must have k_eq_v"

    # ------------------------------------------------------------------
    # QKV behaviour: V should come from raw K for full-attention k_eq_v layers
    # ------------------------------------------------------------------

    def test_k_eq_v_value_uses_pre_k_norm_key(self):
        """For k_eq_v full-attention layers, V = v_norm(raw K), matching HF."""
        config = self._megatron_config()
        full_attn = self._build_attn(config, layer_number=2)

        batch, seq = 2, 5
        hidden = torch.randn(seq, batch, config.hidden_size, device="cuda")

        with torch.no_grad():
            q, k, v = full_attn.get_query_key_value_tensors(hidden)
            _q_raw_path, _k_normed, raw_k = full_attn._get_k_eq_v_query_key_value_tensors(hidden)

        raw_kf = raw_k.float()
        v_expected = (
            raw_kf * torch.pow(raw_kf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        ).to(raw_k)
        torch.testing.assert_close(v, v_expected, atol=1e-6, rtol=1e-6,
                                   msg="V must equal v_norm(raw K) for k_eq_v layer")

        kf = k.float()
        post_norm_expected = (
            kf * torch.pow(kf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        ).to(k)
        assert not torch.allclose(v, post_norm_expected), (
            "V must not use post-k_norm K for k_eq_v"
        )

    def test_sliding_layer_v_differs_from_k(self):
        """Sliding layers should NOT apply k_eq_v; V is from its own projection."""
        config = self._megatron_config()
        sliding_attn = self._build_attn(config, layer_number=1)

        batch, seq = 2, 5
        hidden = torch.randn(seq, batch, config.hidden_size, device="cuda")

        with torch.no_grad():
            q, k, v = sliding_attn.get_query_key_value_tensors(hidden)

        # In general K ≠ V for a randomly initialized sliding layer
        assert not torch.allclose(k, v), "Sliding layer must NOT force V == K"

    # ------------------------------------------------------------------
    # Weight loading: V rows in fused QKV must be zero
    # ------------------------------------------------------------------

    def test_loader_zeros_v_rows_for_k_eq_v_full_attention_layer(self):
        """Loader must zero the V rows of linear_qkv for k_eq_v full-attention layers."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        from tools.checkpoint.loader_gemma4_hf import _set_layer_state

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(4010)
        hf_layers = [
            modeling_gemma4.Gemma4TextDecoderLayer(hf_config, layer_idx=i).cuda()
            for i in range(2)
        ]
        megatron_layers = [
            build_module(
                get_gemma4_layer_spec(megatron_config),
                config=megatron_config,
                layer_number=i + 1,
            ).cuda()
            for i in range(2)
        ]

        import types
        model = types.SimpleNamespace(
            decoder=types.SimpleNamespace(layers=megatron_layers)
        )
        hf_model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=hf_layers)
        )
        loader_args = types.SimpleNamespace(
            num_layers=2,
            num_attention_heads=hf_config.num_attention_heads,
            group_query_attention=True,
            num_query_groups=hf_config.num_key_value_heads,
            num_global_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            global_kv_channels=hf_config.global_head_dim,
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                1 if lt == "sliding_attention" else 0 for lt in hf_config.layer_types
            ],
            untie_embeddings_and_output_weights=True,
            num_kv_shared_layers=0,
            attention_k_eq_v=True,
            enable_moe_block=False,
        )

        for i in range(2):
            _set_layer_state(loader_args, model, hf_model, i)

        # Layer 1 is full-attention with k_eq_v: V rows must be zero
        qkv_w = megatron_layers[1].self_attention.linear_qkv.weight.data
        num_kv = hf_config.num_key_value_heads
        head_dim = hf_config.global_head_dim  # full-attention uses global_head_dim
        num_q_per_group = hf_config.num_attention_heads // num_kv
        group_size = (num_q_per_group + 2) * head_dim
        for g in range(num_kv):
            v_rows = qkv_w[
                g * group_size + (num_q_per_group + 1) * head_dim:
                (g + 1) * group_size
            ]
            assert v_rows.abs().max() == 0.0, (
                f"V rows must be zero for k_eq_v full-attention layer (group {g})"
            )

        # K rows must be non-zero (from real k_proj weights)
        k_rows = qkv_w[num_q_per_group * head_dim: (num_q_per_group + 1) * head_dim]
        assert k_rows.abs().max() > 0, "K rows must be non-zero for k_eq_v layer"

        # Layer 0 (sliding): k_eq_v does NOT apply → V rows can be non-zero
        qkv_w0 = megatron_layers[0].self_attention.linear_qkv.weight.data
        head_dim0 = hf_config.head_dim  # sliding uses normal head_dim
        group_size0 = (num_q_per_group + 2) * head_dim0
        v0_rows = qkv_w0[
            (num_q_per_group + 1) * head_dim0: group_size0
        ]
        assert v0_rows.abs().max() > 0, "Sliding layer V rows must be non-zero (k_eq_v off)"

    def test_k_eq_v_attention_forward_matches_hf_transformers(self):
        """Full-attention k_eq_v attention output matches HF numerically."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        torch.manual_seed(4020)
        hf_attention = modeling_gemma4.Gemma4TextAttention(
            hf_config,
            layer_idx=1,
        ).cuda().eval()
        megatron_attention = self._build_attn(megatron_config, layer_number=2)

        head_dim = hf_attention.head_dim
        num_attention_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        fused_qkv = TestGemma4HFNumericalParity._fuse_qkv_gqa(
            hf_attention.q_proj.weight,
            hf_attention.k_proj.weight,
            torch.zeros_like(hf_attention.k_proj.weight),
            num_attention_heads,
            num_kv_heads,
            head_dim,
        )
        megatron_attention.linear_qkv.weight.data.copy_(fused_qkv)
        megatron_attention.q_layernorm.weight.data.copy_(hf_attention.q_norm.weight)
        megatron_attention.k_layernorm.weight.data.copy_(hf_attention.k_norm.weight)
        megatron_attention.linear_proj.weight.data.copy_(hf_attention.o_proj.weight)

        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden_bsh = torch.randn(batch, seq, hf_config.hidden_size, device="cuda")
        hidden_sbh = hidden_bsh.transpose(0, 1).contiguous()
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        hf_mask = TestGemma4HFNumericalParity._hf_additive_attention_mask(seq, False)
        megatron_pos_emb = megatron_rope(seq)[1]

        with torch.no_grad():
            hf_position_embeddings = hf_rope(
                hidden_bsh,
                position_ids,
                layer_type=hf_attention.layer_type,
            )
            hf_output, _ = hf_attention(
                hidden_states=hidden_bsh,
                position_embeddings=hf_position_embeddings,
                attention_mask=hf_mask,
                shared_kv_states={},
            )
            megatron_output, megatron_bias = megatron_attention(
                hidden_states=hidden_sbh,
                attention_mask=None,
                rotary_pos_emb=(megatron_pos_emb, megatron_pos_emb),
            )

        assert megatron_bias is None
        megatron_output = megatron_output.transpose(0, 1).contiguous()
        torch.testing.assert_close(megatron_output, hf_output, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Forward sanity: finite output with k_eq_v enabled
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_k_eq_v_forward_produces_finite_output(self, layer_number):
        """Forward through k_eq_v layers (both sliding and full) stays finite."""
        config = self._megatron_config()
        layer = build_module(
            get_gemma4_layer_spec(config),
            config=config,
            layer_number=layer_number,
        ).cuda().eval()

        megatron_rope = Gemma4RotaryEmbedding(
            config=config,
            rotary_percent=1.0,
            use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden = torch.randn(seq, batch, config.hidden_size, device="cuda")

        with torch.no_grad():
            out, _ = layer(hidden, attention_mask=None, rotary_pos_emb=megatron_rope(seq))

        assert out.shape == hidden.shape
        assert torch.isfinite(out).all(), f"k_eq_v layer {layer_number} produced non-finite output"


# ---------------------------------------------------------------------------
# Step 4b: Inference / KV-cache mode tests
# ---------------------------------------------------------------------------


class TestGemma4InferenceMode:
    """Static inference-context coverage for Gemma4 GPTModel."""

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(4501)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_gpt_model(self, **cfg_overrides):
        from megatron.core.models.gpt.gpt_model import GPTModel

        defaults = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            scale_embeddings_by_hidden_size=True,
            use_cpu_initialization=True,
        )
        defaults.update(cfg_overrides)
        extra_fields = {}
        for field in (
            "num_kv_shared_layers",
            "attention_k_eq_v",
            "enable_moe_block",
            "num_experts",
            "moe_intermediate_size",
            "top_k_experts",
        ):
            if field in defaults:
                extra_fields[field] = defaults.pop(field)
        config = _make_gemma4_config(**defaults)
        for field, value in extra_fields.items():
            setattr(config, field, value)
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gemma4_layer_spec(config),
            vocab_size=128,
            max_sequence_length=16,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda().eval()
        return model

    @staticmethod
    def _ids(batch: int, seq: int):
        input_ids = torch.arange(batch * seq, device="cuda", dtype=torch.long).view(batch, seq)
        input_ids = input_ids.remainder(128)
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        return input_ids, position_ids

    def test_inference_context_kv_cache_fills_incrementally(self):
        """Static KV cache is allocated on prefill and extended during decode."""
        from megatron.core.inference.contexts import StaticInferenceContext

        model = self._build_gpt_model()
        batch, prompt_len = 2, 5
        input_ids, position_ids = self._ids(batch, prompt_len + 1)
        context = StaticInferenceContext(
            max_batch_size=batch,
            max_sequence_length=prompt_len + 1,
        )

        with torch.no_grad():
            context.enable_prefill_mode()
            model(
                input_ids=input_ids[:, :prompt_len],
                position_ids=position_ids[:, :prompt_len],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )

        assert set(context.key_value_memory_dict.keys()) == {1, 2}
        cached_after_prefill = {
            layer_number: (k.clone(), v.clone())
            for layer_number, (k, v) in context.key_value_memory_dict.items()
        }

        for layer_number, (key_cache, value_cache) in cached_after_prefill.items():
            assert key_cache.shape[0] == prompt_len + 1
            assert key_cache.shape[1] == batch
            assert value_cache.shape[0] == prompt_len + 1
            assert value_cache.shape[1] == batch
            assert torch.isfinite(key_cache[:prompt_len]).all()
            assert torch.isfinite(value_cache[:prompt_len]).all()
            assert key_cache[:prompt_len].abs().max() > 0, f"empty K cache for layer {layer_number}"
            assert value_cache[:prompt_len].abs().max() > 0, f"empty V cache for layer {layer_number}"

        with torch.no_grad():
            context.sequence_len_offset = prompt_len
            context.enable_decode_mode()
            model(
                input_ids=input_ids[:, prompt_len:],
                position_ids=position_ids[:, prompt_len:],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )

        for layer_number, (key_cache, value_cache) in context.key_value_memory_dict.items():
            key_before, value_before = cached_after_prefill[layer_number]
            torch.testing.assert_close(
                key_cache[:prompt_len],
                key_before[:prompt_len],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                value_cache[:prompt_len],
                value_before[:prompt_len],
                atol=0,
                rtol=0,
            )
            assert torch.isfinite(key_cache[prompt_len]).all()
            assert torch.isfinite(value_cache[prompt_len]).all()
            assert key_cache[prompt_len].abs().max() > 0
            assert value_cache[prompt_len].abs().max() > 0

    def test_prefill_then_decode_matches_full_forward(self):
        """Cached one-token decode logits match the same token in a full forward."""
        from megatron.core.inference.contexts import StaticInferenceContext

        model = self._build_gpt_model()
        batch, seq = 2, 6
        input_ids, position_ids = self._ids(batch, seq)

        with torch.no_grad():
            full_logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        context = StaticInferenceContext(max_batch_size=batch, max_sequence_length=seq)
        with torch.no_grad():
            context.enable_prefill_mode()
            model(
                input_ids=input_ids[:, :-1],
                position_ids=position_ids[:, :-1],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )
            context.sequence_len_offset = seq - 1
            context.enable_decode_mode()
            decode_logits = model(
                input_ids=input_ids[:, -1:],
                position_ids=position_ids[:, -1:],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )

        torch.testing.assert_close(
            decode_logits,
            full_logits[:, -1:, :],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_shared_kv_cache_reused_across_layers(self):
        """Shared layer cache receives the same K/V tensors as its source layer."""
        from megatron.core.inference.contexts import StaticInferenceContext
        from megatron.core.models.gpt.gemma4_layer_specs import wire_gemma4_kv_sharing

        model = self._build_gpt_model(
            num_layers=3,
            window_attn_skip_freq=[1, 0, 0],
            num_kv_shared_layers=1,
        )
        wire_gemma4_kv_sharing(model)

        batch, prompt_len = 2, 5
        input_ids, position_ids = self._ids(batch, prompt_len + 1)
        context = StaticInferenceContext(
            max_batch_size=batch,
            max_sequence_length=prompt_len + 1,
        )

        with torch.no_grad():
            context.enable_prefill_mode()
            model(
                input_ids=input_ids[:, :prompt_len],
                position_ids=position_ids[:, :prompt_len],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )

            context.sequence_len_offset = prompt_len
            context.enable_decode_mode()
            model(
                input_ids=input_ids[:, prompt_len:],
                position_ids=position_ids[:, prompt_len:],
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )

        source_key, source_value = context.key_value_memory_dict[2]
        shared_key, shared_value = context.key_value_memory_dict[3]
        torch.testing.assert_close(
            shared_key[: prompt_len + 1],
            source_key[: prompt_len + 1],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            shared_value[: prompt_len + 1],
            source_value[: prompt_len + 1],
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# Step 4c: Mixed precision sanity tests
# ---------------------------------------------------------------------------


class TestGemma4MixedPrecision:
    """bf16/fp16 forward coverage for Gemma4 feature combinations."""

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(4701)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_model(self, dtype: torch.dtype, **cfg_overrides):
        from megatron.core.models.gpt.gpt_model import GPTModel

        defaults = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            scale_embeddings_by_hidden_size=True,
            use_cpu_initialization=True,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
        )
        defaults.update(cfg_overrides)
        extra_fields = {}
        for field in (
            "num_kv_shared_layers",
            "attention_k_eq_v",
            "enable_moe_block",
            "num_experts",
            "moe_intermediate_size",
            "top_k_experts",
        ):
            if field in defaults:
                extra_fields[field] = defaults.pop(field)
        config = _make_gemma4_config(**defaults)
        for field, value in extra_fields.items():
            setattr(config, field, value)

        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gemma4_layer_spec(config),
            vocab_size=128,
            max_sequence_length=16,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda().to(dtype).eval()
        return model

    @staticmethod
    def _ids(batch: int = 2, seq: int = 6):
        input_ids = torch.arange(batch * seq, device="cuda", dtype=torch.long).view(batch, seq)
        input_ids = input_ids.remainder(128)
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        return input_ids, position_ids

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_forward_stays_finite(self, dtype):
        """bf16/fp16 GPTModel forward produces finite logits."""
        model = self._build_model(dtype)
        input_ids, position_ids = self._ids()

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        assert logits.dtype == dtype
        assert torch.isfinite(logits).all()

# ---------------------------------------------------------------------------
# Step 5: MoE block parity tests
# ---------------------------------------------------------------------------


class TestGemma4MoEBlockParity:
    """Tests for the MoE block (enable_moe_block=True).

    All layers in the test config have the MoE block enabled.  The dense MLP
    and sparse expert outputs are combined via 3 extra layernorms and added to
    the pre-MLP residual.  Full HF forward parity is achievable here because
    both implementations use the same token-independent routing logic.
    """

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(5001)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    _NUM_EXPERTS = 2
    _MOE_INTERMEDIATE = 32
    _TOP_K = 1

    def _hf_config(self):
        configuration_gemma4 = pytest.importorskip(
            "transformers.models.gemma4.configuration_gemma4"
        )
        cfg = configuration_gemma4.Gemma4TextConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=0,
            max_position_embeddings=32,
            final_logit_softcapping=None,
            tie_word_embeddings=False,
            use_cache=False,
            enable_moe_block=True,
            num_experts=self._NUM_EXPERTS,
            moe_intermediate_size=self._MOE_INTERMEDIATE,
            top_k_experts=self._TOP_K,
        )
        cfg._attn_implementation = "eager"
        return cfg

    def _megatron_config(self):
        cfg = _make_gemma4_config(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            use_cpu_initialization=True,
        )
        cfg.enable_moe_block = True
        cfg.num_experts = self._NUM_EXPERTS
        cfg.moe_intermediate_size = self._MOE_INTERMEDIATE
        cfg.top_k_experts = self._TOP_K
        return cfg

    def _build_megatron_layer(self, config, layer_number):
        return build_module(
            get_gemma4_layer_spec(config), config=config, layer_number=layer_number
        ).cuda()

    def _copy_layer_moe(self, megatron_layer, hf_layer, hf_config, layer_idx):
        """Copy weights from HF to Megatron (uses loader _set_layer_state)."""
        import types

        from tools.checkpoint.loader_gemma4_hf import _set_layer_state

        model = types.SimpleNamespace(
            decoder=types.SimpleNamespace(layers=[None, None])
        )
        hf_model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[None, None])
        )
        model.decoder.layers[layer_idx] = megatron_layer
        hf_model.model.layers[layer_idx] = hf_layer

        loader_args = types.SimpleNamespace(
            num_layers=2,
            num_attention_heads=hf_config.num_attention_heads,
            group_query_attention=True,
            num_query_groups=hf_config.num_key_value_heads,
            num_global_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            global_kv_channels=hf_config.global_head_dim,
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                1 if lt == "sliding_attention" else 0 for lt in hf_config.layer_types
            ],
            untie_embeddings_and_output_weights=True,
            num_kv_shared_layers=0,
            attention_k_eq_v=False,
            enable_moe_block=True,
        )
        _set_layer_state(loader_args, model, hf_model, layer_idx)

    # ------------------------------------------------------------------
    # Module creation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("enable", [False, True])
    def test_moe_modules_created_iff_enabled(self, enable):
        """moe_router and moe_experts must exist iff enable_moe_block=True."""
        config = _make_gemma4_config(
            num_layers=2, hidden_size=64, num_attention_heads=4,
            num_query_groups=2, kv_channels=16, ffn_hidden_size=128,
            use_cpu_initialization=True,
        )
        if enable:
            config.enable_moe_block = True
            config.num_experts = 2
            config.moe_intermediate_size = 32
            config.top_k_experts = 1
        else:
            config.enable_moe_block = False

        layer = build_module(
            get_gemma4_layer_spec(config), config=config, layer_number=1
        ).cuda()

        if enable:
            assert layer.moe_router is not None, "moe_router must exist when enabled"
            assert layer.moe_experts is not None, "moe_experts must exist when enabled"
            assert layer.post_feedforward_layernorm_1 is not None
            assert layer.post_feedforward_layernorm_2 is not None
            assert layer.pre_feedforward_layernorm_2 is not None
        else:
            assert layer.moe_router is None, "moe_router must be None when disabled"
            assert layer.moe_experts is None, "moe_experts must be None when disabled"

    # ------------------------------------------------------------------
    # Router and experts shapes
    # ------------------------------------------------------------------

    def test_moe_router_output_shapes(self):
        """Gemma4MoERouter output shapes match [tokens, num_experts] / [tokens, top_k]."""
        from megatron.core.models.gpt.gemma4_layer_specs import Gemma4MoERouter

        config = self._megatron_config()
        router = Gemma4MoERouter(config).cuda().eval()

        tokens = 10
        hidden = torch.randn(tokens, config.hidden_size, device="cuda")
        with torch.no_grad():
            router_probs, top_k_weights, top_k_index = router(hidden)

        assert router_probs.shape == (tokens, self._NUM_EXPERTS)
        assert top_k_weights.shape == (tokens, self._TOP_K)
        assert top_k_index.shape == (tokens, self._TOP_K)

        # Each row of router_probs should sum to ~1
        torch.testing.assert_close(
            router_probs.sum(dim=-1),
            torch.ones(tokens, device="cuda"),
            atol=1e-5, rtol=1e-5,
        )

    def test_moe_experts_output_shapes(self):
        """Gemma4MoEExperts preserves [tokens, hidden_size] shape."""
        from megatron.core.models.gpt.gemma4_layer_specs import Gemma4MoEExperts, Gemma4MoERouter

        config = self._megatron_config()
        router = Gemma4MoERouter(config).cuda().eval()
        experts = Gemma4MoEExperts(config).cuda().eval()

        tokens = 8
        hidden = torch.randn(tokens, config.hidden_size, device="cuda")

        with torch.no_grad():
            _, top_k_weights, top_k_index = router(hidden)
            out = experts(hidden, top_k_index, top_k_weights)

        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("layer_idx,layer_number", [(0, 1), (1, 2)])
    def test_moe_weight_loading_matches_hf(self, layer_idx, layer_number):
        """Loader correctly copies router + expert weights from HF."""
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        # Reset ALL RNG states (including Megatron's expert-parallel tracker)
        # before model creation so rank-specific tracker state does not bleed
        # into nn.init.normal_ calls via CPU-RNG consumption differences.
        torch.manual_seed(5010)
        model_parallel_cuda_manual_seed(5010)
        hf_layer = modeling_gemma4.Gemma4TextDecoderLayer(
            hf_config, layer_idx=layer_idx
        ).cuda()
        megatron_layer = self._build_megatron_layer(megatron_config, layer_number)
        self._copy_layer_moe(megatron_layer, hf_layer, hf_config, layer_idx)

        # Router: scale, proj, per_expert_scale
        torch.testing.assert_close(
            megatron_layer.moe_router.scale,
            hf_layer.router.scale,
            msg="Router scale mismatch",
        )
        torch.testing.assert_close(
            megatron_layer.moe_router.proj.weight,
            hf_layer.router.proj.weight,
            msg="Router proj.weight mismatch",
        )
        torch.testing.assert_close(
            megatron_layer.moe_router.per_expert_scale,
            hf_layer.router.per_expert_scale,
            msg="Router per_expert_scale mismatch",
        )

        # Expert weights (3D tensors)
        torch.testing.assert_close(
            megatron_layer.moe_experts.gate_up_proj,
            hf_layer.experts.gate_up_proj,
            msg="Expert gate_up_proj mismatch",
        )
        torch.testing.assert_close(
            megatron_layer.moe_experts.down_proj,
            hf_layer.experts.down_proj,
            msg="Expert down_proj mismatch",
        )

        # Extra norms
        torch.testing.assert_close(
            megatron_layer.post_feedforward_layernorm_1.weight,
            hf_layer.post_feedforward_layernorm_1.weight,
        )
        torch.testing.assert_close(
            megatron_layer.post_feedforward_layernorm_2.weight,
            hf_layer.post_feedforward_layernorm_2.weight,
        )
        torch.testing.assert_close(
            megatron_layer.pre_feedforward_layernorm_2.weight,
            hf_layer.pre_feedforward_layernorm_2.weight,
        )

    # ------------------------------------------------------------------
    # Full forward parity with HF
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("layer_idx,layer_number", [(0, 1), (1, 2)])
    def test_moe_layer_forward_matches_hf(self, layer_idx, layer_number):
        """Full MoE decoder-layer forward must be numerically identical to HF.

        The token-routing in Gemma4MoEExperts is commutative w.r.t. batch/seq
        ordering, so seq-first [s,b,h] and batch-first [b,s,h] layouts yield
        identical per-token results after reshaping.
        """
        modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

        hf_config = self._hf_config()
        megatron_config = self._megatron_config()

        # Force deterministic CUBLAS/cuDNN to avoid algorithm-selection differences
        # when this test runs after TestGemma4EndToEnd's distributed training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        torch.manual_seed(5020)
        torch.cuda.manual_seed_all(5020)
        hf_layer = modeling_gemma4.Gemma4TextDecoderLayer(
            hf_config, layer_idx=layer_idx
        ).cuda().eval()
        megatron_layer = self._build_megatron_layer(megatron_config, layer_number).eval()
        self._copy_layer_moe(megatron_layer, hf_layer, hf_config, layer_idx)

        hf_rope = modeling_gemma4.Gemma4TextRotaryEmbedding(hf_config).cuda()
        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config, rotary_percent=1.0, use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        # Generate on CPU (deterministic under torch.manual_seed) then move to CUDA,
        # so the input is insulated from CUDA RNG consumed by layer creation.
        hidden_bsh = torch.randn(batch, seq, hf_config.hidden_size).cuda()
        hidden_sbh = hidden_bsh.transpose(0, 1).contiguous()
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)

        is_sliding = hf_layer.self_attn.is_sliding
        layer_type = hf_layer.self_attn.layer_type
        # Build an additive causal mask for HF
        from megatron.core.transformer.utils import (
            get_default_causal_mask,
            get_sliding_window_causal_mask,
        )
        if is_sliding:
            bool_mask = get_sliding_window_causal_mask(seq, seq, (3, 0))
        else:
            bool_mask = get_default_causal_mask(seq)
        hf_mask = torch.zeros(1, 1, seq, seq, device="cuda").masked_fill(
            bool_mask.view(1, 1, seq, seq), -10000.0
        )

        with torch.no_grad():
            hf_pos_emb = hf_rope(hidden_bsh, position_ids, layer_type=layer_type)
            hf_out = hf_layer(
                hidden_states=hidden_bsh,
                per_layer_input=None,
                shared_kv_states={},
                position_embeddings=hf_pos_emb,
                attention_mask=hf_mask,
                position_ids=position_ids,
            )
            # Megatron rope embs: (sliding_emb, full_emb) tuple
            rope_embs = megatron_rope(seq)
            megatron_out, _ = megatron_layer(
                hidden_states=hidden_sbh,
                attention_mask=None,
                rotary_pos_emb=rope_embs,
            )

        megatron_out_bsh = megatron_out.transpose(0, 1).contiguous()
        assert megatron_out_bsh.shape == hf_out.shape

        torch.testing.assert_close(
            megatron_out_bsh, hf_out, atol=1e-3, rtol=1e-3,
            msg=f"MoE layer {layer_idx} forward mismatch vs HF",
        )

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_moe_layer_backward_produces_finite_gradients(self, layer_number):
        """Backward pass through MoE layer must produce finite gradients.

        Uses Megatron's own deterministic CPU initialization so the test does not
        depend on the HF transformers package and is not sensitive to specific
        random weight draws that can produce NaN via HF's Kaiming-uniform init.
        """
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")

        megatron_config = self._megatron_config()

        torch.manual_seed(5030)
        torch.cuda.manual_seed_all(5030)
        megatron_layer = self._build_megatron_layer(megatron_config, layer_number)

        megatron_rope = Gemma4RotaryEmbedding(
            config=megatron_config, rotary_percent=1.0, use_cpu_initialization=True,
        )

        batch, seq = 2, 5
        hidden_sbh = (
            torch.randn(seq, batch, megatron_config.hidden_size).cuda().requires_grad_(True)
        )

        out, _ = megatron_layer(
            hidden_states=hidden_sbh,
            attention_mask=None,
            rotary_pos_emb=megatron_rope(seq),
        )
        out.sum().backward()

        assert hidden_sbh.grad is not None, "No gradient for input hidden states"
        assert torch.isfinite(hidden_sbh.grad).all(), "Non-finite input gradients"

        for name, param in megatron_layer.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for parameter {name}"
                )


# ---------------------------------------------------------------------------
# Step 6: Tensor-parallel smoke tests
# ---------------------------------------------------------------------------
# NOTE: TP=2 tests come after MoE parity tests to avoid CUDA kernel-cache
# contamination: TP=2 teardown can leave cuBLAS in a state that changes
# algorithm selection for the subsequent TP=1 MoE forward, breaking the
# atol=1e-5 parity assertion.
# ---------------------------------------------------------------------------


class TestGemma4TensorParallelism:
    """TP>1 coverage. These tests require launching pytest with WORLD_SIZE>=2."""

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        if Utils.world_size < 2:
            pytest.skip("TP>1 tests require WORLD_SIZE>=2, e.g. torchrun --nproc_per_node=2")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(5101)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_model(self, **cfg_overrides):
        from megatron.core.models.gpt.gpt_model import GPTModel

        defaults = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            global_kv_channels=32,
            num_global_query_groups=2,
            ffn_hidden_size=128,
            window_size=(3, 0),
            window_attn_skip_freq=2,
            activation_func=_gelu_pytorch_tanh,
            sliding_window_rope_base=10000.0,
            full_attention_rope_base=1000000.0,
            full_attention_rope_partial_factor=0.25,
            scale_embeddings_by_hidden_size=True,
            use_cpu_initialization=True,
        )
        defaults.update(cfg_overrides)
        extra_fields = {}
        for field in (
            "attention_k_eq_v",
            "enable_moe_block",
            "num_experts",
            "moe_intermediate_size",
            "top_k_experts",
        ):
            if field in defaults:
                extra_fields[field] = defaults.pop(field)
        config = _make_gemma4_config(**defaults)
        for field, value in extra_fields.items():
            setattr(config, field, value)

        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gemma4_layer_spec(config),
            vocab_size=128,
            max_sequence_length=16,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda().eval()
        return model

    @staticmethod
    def _ids(batch: int = 2, seq: int = 6):
        input_ids = torch.arange(batch * seq, device="cuda", dtype=torch.long).view(batch, seq)
        input_ids = input_ids.remainder(128)
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
        return input_ids, position_ids

    def test_tp2_forward_produces_finite_logits(self):
        """Gemma4 GPTModel runs under TP=2 and gathers output logits."""
        model = self._build_model()
        input_ids, position_ids = self._ids()

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        assert logits.shape == (2, 6, 128)
        assert torch.isfinite(logits).all()

    def test_tp2_moe_forward_routes_finitely(self):
        """MoE path runs under TP=2 and produces finite logits."""
        model = self._build_model(
            enable_moe_block=True,
            num_experts=3,
            moe_intermediate_size=32,
            top_k_experts=2,
        )
        input_ids, position_ids = self._ids()

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

        assert logits.shape == (2, 6, 128)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Step 7: End-to-end training / checkpoint round-trip
# ---------------------------------------------------------------------------


class TestGemma4EndToEnd:
    """End-to-end GPTModel tests: forward, backward, loss decrease, checkpoint."""

    def setup_method(self):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(7)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_gpt_model(self, **cfg_overrides):
        """Build a small Gemma4 GPTModel on CUDA."""
        import torch.nn.functional as F

        from megatron.core.models.gpt.gpt_model import GPTModel

        config = _make_gemma4_config(**cfg_overrides)
        spec = get_gemma4_layer_spec(config)
        model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=256,
            max_sequence_length=16,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type='rope',
        ).cuda()
        return model

    def _dummy_batch(self, batch: int = 2, seq: int = 8):
        input_ids = torch.randint(0, 256, (batch, seq), device='cuda')
        position_ids = torch.arange(seq, device='cuda').unsqueeze(0).expand(batch, -1)
        labels = torch.randint(0, 256, (batch, seq), device='cuda')
        return input_ids, position_ids, labels

    # ------------------------------------------------------------------
    # forward tests
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_forward_produces_correct_shape(self):
        """GPTModel forward → logits with shape [batch, seq, vocab]."""
        model = self._build_gpt_model()
        input_ids, position_ids, _ = self._dummy_batch()
        with torch.no_grad():
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        assert logits.shape == (2, 8, 256)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_forward_produces_finite_logits(self):
        """Logits must not contain NaN or Inf."""
        model = self._build_gpt_model()
        input_ids, position_ids, _ = self._dummy_batch()
        with torch.no_grad():
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_forward_with_ple_produces_finite_logits(self):
        """Forward with per_layer_embed_dim > 0 must also produce finite logits."""
        model = self._build_gpt_model(per_layer_embed_vocab_size=256, per_layer_embed_dim=32)
        input_ids, position_ids, _ = self._dummy_batch()
        with torch.no_grad():
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        assert torch.isfinite(logits).all()

    # ------------------------------------------------------------------
    # backward tests
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_backward_completes_without_error(self):
        """loss.backward() must complete and produce non-None gradients."""
        import torch.nn.functional as F

        model = self._build_gpt_model()
        input_ids, position_ids, labels = self._dummy_batch()

        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        loss = F.cross_entropy(logits.reshape(-1, 256), labels.reshape(-1))
        loss.backward()

        # At least one parameter must have a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients were computed"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gradients_are_finite(self):
        """All gradients must be finite (no NaN/Inf explosion)."""
        import torch.nn.functional as F

        model = self._build_gpt_model()
        input_ids, position_ids, labels = self._dummy_batch()

        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        loss = F.cross_entropy(logits.reshape(-1, 256), labels.reshape(-1))
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient in parameter '{name}'"
                )

    # ------------------------------------------------------------------
    # loss decrease test
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_loss_decreases_after_optimizer_step(self):
        """A few optimizer steps on the same tiny batch should reduce the training loss."""
        import torch.nn.functional as F

        model = self._build_gpt_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        torch.manual_seed(99)
        input_ids, position_ids, labels = self._dummy_batch()

        def compute_loss():
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            return F.cross_entropy(logits.reshape(-1, 256), labels.reshape(-1))

        loss_before = compute_loss().item()
        for _ in range(5):
            optimizer.zero_grad()
            compute_loss().backward()
            optimizer.step()
        loss_after = compute_loss().item()

        assert loss_after < loss_before, (
            f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    # ------------------------------------------------------------------
    # checkpoint round-trip tests
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_checkpoint_state_dict_round_trip(self):
        """Save and restore state_dict; all parameter values must match."""
        model = self._build_gpt_model()

        # _extra_state keys can be None when FP8/quantization is not in use.
        state = {k: v.clone() for k, v in model.state_dict().items() if v is not None}

        # Perturb parameters to simulate training
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Restore
        model.load_state_dict(state, strict=False)

        for name, restored in model.state_dict().items():
            if name not in state:
                continue
            torch.testing.assert_close(
                restored, state[name],
                msg=f"Parameter '{name}' not restored correctly",
            )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_checkpoint_logits_match_after_reload(self):
        """Logits from a reloaded model match the original model's logits."""
        model = self._build_gpt_model()
        model.eval()

        input_ids, position_ids, _ = self._dummy_batch(batch=1, seq=4)

        with torch.no_grad():
            logits_before = model(
                input_ids=input_ids, position_ids=position_ids, attention_mask=None
            ).clone()

        # _extra_state keys can be None when FP8/quantization is not in use.
        state = {k: v.clone() for k, v in model.state_dict().items() if v is not None}

        # Perturb then reload
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            logits_after = model(
                input_ids=input_ids, position_ids=position_ids, attention_mask=None
            )

        torch.testing.assert_close(logits_before, logits_after, atol=1e-6, rtol=1e-5)
