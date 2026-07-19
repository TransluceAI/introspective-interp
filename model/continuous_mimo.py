"""
MiMo-7B adapter for introspective-interp framework.

This module provides the ContinuousMiMo class that integrates Xiaomi's MiMo-7B
reasoning model into the introspective-interp framework for self-explanation tasks.

MiMo-7B features:
- 7B parameter reasoning-focused model based on Qwen2 architecture
- Multiple-Token Prediction (MTP) auxiliary objective
- Supports vLLM and SGLang deployment
- Achieves 95.8% on MATH500, 68.2% on AIME 2024

Usage:
    model = ContinuousMiMo.from_pretrained(
        "XiaomiMiMo/MiMo-7B-RL",
        trust_remote_code=True,
        ...
    )
"""

from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import Cache

from .continuous_base import ContinuousCausalLMBase, ContinuousCausalLMOutputWithPast


class ContinuousMiMo(ContinuousCausalLMBase, nn.Module):
    """
    Extension of MiMo-7B that allows for intervention in the hidden states
    and attention head outputs during the forward pass.

    MiMo uses a Qwen2-based architecture with Multiple-Token Prediction (MTP),
    so we load it via AutoModelForCausalLM with trust_remote_code=True
    and wrap it for continuous token support.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        batch_size: int = 24,
        special_tokens_ids: Dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = False,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        nn.Module.__init__(self)

        # Store config for compatibility
        self.config = config

        # Initialize the base model placeholder
        self._base_model = None
        self._model_initialized = False

        # Store initialization parameters
        self._init_params = {
            "batch_size": batch_size,
            "special_tokens_ids": special_tokens_ids,
            "subject_embed_dim": subject_embed_dim,
            "use_embed_proj": use_embed_proj,
            "target_model_layers": target_model_layers,
        }

        # State for hidden state interventions
        self._intervention_hidden_states = None
        self._intervention_active = False
        self._intervention_is_batched = False

        # State for attention head interventions
        self._head_intervention_layer = None
        self._head_intervention_indices_positions = None
        self._head_intervention_active = False
        self._head_intervention_is_batched = False
        self._head_patching_handles = []

        # State for attribution patching
        self._attribution_active = False
        self._attribution_hooks = []
        self._attribution_gradients = {}

    def _init_continuous_base(self):
        """Initialize ContinuousCausalLMBase after model is loaded."""
        ContinuousCausalLMBase.__init__(
            self,
            batch_size=self._init_params["batch_size"],
            special_tokens_ids=self._init_params["special_tokens_ids"],
            subject_embed_dim=self._init_params["subject_embed_dim"],
            use_embed_proj=self._init_params["use_embed_proj"],
            target_model_layers=self._init_params["target_model_layers"],
        )
        self._model_initialized = True

    @property
    def model(self):
        """Access the underlying transformer model.

        MiMo (like Qwen2) has the transformer layers under .model attribute.
        """
        if self._base_model is None:
            raise RuntimeError("Model not initialized. Call from_pretrained first.")
        # MiMo/Qwen2 models have transformer under .model
        if hasattr(self._base_model, 'model'):
            return self._base_model.model
        return self._base_model

    @property
    def lm_head(self):
        """Access the language model head."""
        if self._base_model is None:
            raise RuntimeError("Model not initialized. Call from_pretrained first.")
        return self._base_model.lm_head

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        batch_size: int = 24,
        special_tokens_ids: Dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = False,
        target_model_layers: int | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        """
        Load a MiMo model from pretrained weights.

        Args:
            model_path: HuggingFace model path (e.g., "XiaomiMiMo/MiMo-7B-RL")
            batch_size: Batch size for training
            special_tokens_ids: Dictionary of special token IDs
            subject_embed_dim: Dimension of subject embeddings
            use_embed_proj: Whether to use embedding projection
            target_model_layers: Number of target model layers
            cache_dir: Cache directory for model weights
            **kwargs: Additional arguments passed to AutoModelForCausalLM

        Returns:
            ContinuousMiMo: Initialized model wrapper
        """
        # Ensure trust_remote_code is set for MiMo
        kwargs['trust_remote_code'] = True

        if cache_dir is not None:
            kwargs['cache_dir'] = cache_dir

        # Load the config first
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # Create the wrapper instance
        instance = cls(
            config=config,
            batch_size=batch_size,
            special_tokens_ids=special_tokens_ids,
            subject_embed_dim=subject_embed_dim,
            use_embed_proj=use_embed_proj,
            target_model_layers=target_model_layers,
        )

        # Load the actual model
        instance._base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )

        # Now initialize the continuous base with the loaded model
        instance._init_continuous_base()

        return instance

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        inputs_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        labels_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        debug: bool = False,
        extra_args: Optional[list[dict]] = None,
        **kwargs,
    ) -> Union[Tuple, ContinuousCausalLMOutputWithPast]:
        """
        Forward pass with continuous token support.

        inputs_continuous_tokens: List[torch.LongTensor] of dim batch_size x (seq_len)
        labels_continuous_tokens: List[torch.LongTensor] of dim batch_size x (seq_len)
        """
        # Handle both parameter names for logits keeping
        if logits_to_keep == 0 and num_logits_to_keep != 0:
            logits_to_keep = num_logits_to_keep

        return self.shared_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            inputs_continuous_tokens=inputs_continuous_tokens,
            labels_continuous_tokens=labels_continuous_tokens,
            debug=debug,
            extra_args=extra_args,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        """Generate text using the underlying MiMo model.

        MiMo recommends:
        - Empty system prompt
        - Temperature 0.6
        """
        if self._base_model is None:
            raise RuntimeError("Model not initialized. Call from_pretrained first.")
        return self._base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        if self._base_model is None:
            raise RuntimeError("Model not initialized. Call from_pretrained first.")
        self._base_model.save_pretrained(save_directory, **kwargs)

        # Also save embed_projs if they exist
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            import os
            embed_proj_path = os.path.join(save_directory, "embed_projs.pt")
            torch.save(self.embed_projs.state_dict(), embed_proj_path)

        print(f"Saved MiMo model to {save_directory}")

    def train(self, mode: bool = True):
        """Set training mode."""
        if self._base_model is not None:
            self._base_model.train(mode)
        # Call parent train method for loss function setup
        if hasattr(super(), 'train'):
            super().train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        if self._base_model is not None:
            self._base_model.eval()
        if hasattr(super(), 'eval'):
            super().eval()
        return self

    def to(self, *args, **kwargs):
        """Move model to device."""
        if self._base_model is not None:
            self._base_model.to(*args, **kwargs)
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            self.embed_projs.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """Move model to CUDA."""
        if self._base_model is not None:
            self._base_model.cuda(device)
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            self.embed_projs.cuda(device)
        return self

    def cpu(self):
        """Move model to CPU."""
        if self._base_model is not None:
            self._base_model.cpu()
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            self.embed_projs.cpu()
        return self

    def half(self):
        """Convert model to half precision."""
        if self._base_model is not None:
            self._base_model.half()
        return self

    def bfloat16(self):
        """Convert model to bfloat16."""
        if self._base_model is not None:
            self._base_model.bfloat16()
        return self

    def float(self):
        """Convert model to float32."""
        if self._base_model is not None:
            self._base_model.float()
        return self

    def parameters(self, recurse: bool = True):
        """Return model parameters."""
        params = []
        if self._base_model is not None:
            params.extend(self._base_model.parameters(recurse))
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            params.extend(self.embed_projs.parameters(recurse))
        return iter(params)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters."""
        params = []
        if self._base_model is not None:
            for name, param in self._base_model.named_parameters(prefix, recurse):
                params.append((name, param))
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            embed_prefix = f"{prefix}embed_projs." if prefix else "embed_projs."
            for name, param in self.embed_projs.named_parameters('', recurse):
                params.append((f"{embed_prefix}{name}", param))
        return iter(params)

    def state_dict(self, *args, **kwargs):
        """Return state dict."""
        state = {}
        if self._base_model is not None:
            state.update(self._base_model.state_dict(*args, **kwargs))
        if hasattr(self, 'embed_projs') and self.embed_projs is not None:
            for i, proj in enumerate(self.embed_projs):
                for name, param in proj.state_dict().items():
                    state[f"embed_projs.{i}.{name}"] = param
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict."""
        # Separate embed_projs from base model state
        base_state = {}
        embed_state = {}
        for key, value in state_dict.items():
            if key.startswith("embed_projs."):
                embed_state[key] = value
            else:
                base_state[key] = value

        if self._base_model is not None and base_state:
            self._base_model.load_state_dict(base_state, strict=strict)

        if hasattr(self, 'embed_projs') and self.embed_projs is not None and embed_state:
            for key, value in embed_state.items():
                parts = key.split(".")
                if len(parts) >= 3:
                    idx = int(parts[1])
                    param_name = ".".join(parts[2:])
                    if idx < len(self.embed_projs):
                        self.embed_projs[idx].load_state_dict({param_name: value}, strict=strict)

    @property
    def device(self):
        """Get model device."""
        if self._base_model is not None:
            return next(self._base_model.parameters()).device
        return torch.device('cpu')

    @property
    def dtype(self):
        """Get model dtype."""
        if self._base_model is not None:
            return next(self._base_model.parameters()).dtype
        return torch.float32

    def get_input_embeddings(self):
        """Get input embeddings layer."""
        if self._base_model is not None and hasattr(self._base_model, 'get_input_embeddings'):
            return self._base_model.get_input_embeddings()
        return None

    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        if self._base_model is not None and hasattr(self._base_model, 'set_input_embeddings'):
            self._base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings layer (lm_head)."""
        if self._base_model is not None and hasattr(self._base_model, 'get_output_embeddings'):
            return self._base_model.get_output_embeddings()
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        if self._base_model is not None and hasattr(self._base_model, 'resize_token_embeddings'):
            return self._base_model.resize_token_embeddings(new_num_tokens)
        return None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        if self._base_model is not None and hasattr(self._base_model, 'gradient_checkpointing_enable'):
            self._base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if self._base_model is not None and hasattr(self._base_model, 'gradient_checkpointing_disable'):
            self._base_model.gradient_checkpointing_disable()

    def __repr__(self):
        return f"ContinuousMiMo(config={self.config})"
