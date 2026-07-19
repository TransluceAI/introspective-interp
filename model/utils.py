import os
import warnings
from typing import Dict, Optional, Tuple

import torch
from model.continuous_gemma3 import ContinuousGemma3ForCausalLM
from model.continuous_gemma2 import ContinuousGemma2ForCausalLM
from model.continuous_qwen import ContinuousQwen3ForCausalLM
from model.continuous_llama import ContinuousLlama
from model.continuous_mimo import ContinuousMiMo
from model.nearest_neighbor import NearestNeighborModel
from model.continuous_peft import ContinuousPeft
from model.self_explanations import SelfExplanationsModel
from peft import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
)
from peft.tuners.tuners_utils import BaseTuner
from peft.utils import _prepare_prompt_learning_config
from transformers import AutoModelForCausalLM, PreTrainedModel


MODEL_TYPE_TO_VANILLA_MODEL_MAPPING = {
    "llama": ContinuousLlama,
    "gemma3": ContinuousGemma3ForCausalLM,
    "gemma2": ContinuousGemma2ForCausalLM,
    "qwen3": ContinuousQwen3ForCausalLM,
    "mimo": ContinuousMiMo,
    "nearest_neighbor": NearestNeighborModel,
    "self_explanations": SelfExplanationsModel,
}


def load_target_model(
    target_model_path: str,
    cache_dir: str,
    use_bf16: bool,
) -> PreTrainedModel:
    """
    Load the target model.
    """
    return AutoModelForCausalLM.from_pretrained(
        target_model_path,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
        cache_dir=cache_dir,
    )


def load_models(
    predictor_model_path: str,
    target_model_path: str | None,
    special_tokens_ids: Optional[Dict[str, int]],
    cache_dir: str,
    train_self: bool,
    use_bf16: bool,
    batch_size: int,
    ckpt_dir: str | None = None,
    do_save: bool = True,
    use_embed_proj: bool = False,
    embed_proj_path: str | None = None,
    predictor_model_type: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedModel, str]:
    """
    Load the target model and predictor model.
    """
    target_model = None
    target_model_layers = None
    if not train_self and target_model_path is not None:
        # Different architecture
        target_model = load_target_model(target_model_path, cache_dir, use_bf16)
        target_model_layers = target_model.config.num_hidden_layers

    if predictor_model_type is None:
        # model_type = predictor_model_type
        predictor_model_type = predictor_model_path
    if predictor_model_type in [
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ]:
        model_type = "gemma3"
    elif predictor_model_type in [
        "google/gemma-2-2b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b",
        "google/gemma-2-9b-it",
    ]:
        model_type = "gemma2"
    elif predictor_model_type in ["Qwen/Qwen3-8B"]:
        model_type = "qwen3"
    elif predictor_model_type in [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
    ]:
        model_type = "llama"
    elif predictor_model_type in [
        "XiaomiMiMo/MiMo-7B-Base",
        "XiaomiMiMo/MiMo-7B-RL",
        "XiaomiMiMo/MiMo-7B-RL-0530",
        "XiaomiMiMo/MiMo-7B-SFT",
        "XiaomiMiMo/MiMo-7B-RL-Zero",
    ] or "MiMo" in predictor_model_type:
        model_type = "mimo"
    else:
        raise ValueError(f"Model {predictor_model_type} not supported for autointerp")

    print("Loading model of type: ", model_type, "from: ", predictor_model_path)

    # Load model
    base_model = MODEL_TYPE_TO_VANILLA_MODEL_MAPPING[model_type].from_pretrained(
        predictor_model_path,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
        batch_size=batch_size,
        special_tokens_ids=special_tokens_ids,
        cache_dir=cache_dir,
        subject_embed_dim=(
            None if target_model is None else target_model.config.hidden_size
        ),
        use_embed_proj=use_embed_proj,
        target_model_layers=target_model_layers,
    )

    # Load embed_proj from embed_proj_path if provided
    # (override existing embed_projs)
    if embed_proj_path is not None and hasattr(base_model, "embed_projs"):
        """
        Load pre-trained embed_proj weights from a file.
        """
        if os.path.exists(embed_proj_path):
            print(f"Loading embed_proj from {embed_proj_path}")
            # Load the alignment model weights and set them to embed_projs
            embed_proj_weights = torch.load(embed_proj_path, map_location="cpu")
            if (
                hasattr(base_model, "embed_projs")
                and base_model.embed_projs is not None
            ):
                # Check if this is a LinearAlignmentModule state dict (has alignments.X.weight keys)
                if any(
                    key.startswith("alignments.") for key in embed_proj_weights.keys()
                ):
                    # Extract the weights from all available alignment layers
                    available_layers = []
                    for key in embed_proj_weights.keys():
                        if key.startswith("alignments.") and key.endswith(".weight"):
                            layer_num = key.split(".")[1]
                            available_layers.append(int(layer_num))

                    if available_layers:
                        print(
                            f"Loading alignment weights for {len(available_layers)} layers"
                        )

                        # Load weights for each available layer into corresponding embed_proj
                        for layer in available_layers:
                            if layer < len(base_model.embed_projs):
                                # Create the expected state dict for nn.Linear (no bias for embed_projs)
                                linear_state_dict = {
                                    "weight": embed_proj_weights[
                                        f"alignments.{layer}.weight"
                                    ].T
                                    # Note: embed_projs have bias=False, so we don't load bias
                                }
                                base_model.embed_projs[layer].load_state_dict(
                                    linear_state_dict
                                )
                                print(f"Loaded alignment weights for layer {layer}")
                            else:
                                print(
                                    f"Warning: Layer {layer} exceeds model layer count, skipping"
                                )

                        print(
                            f"Successfully loaded embed_proj weights for {len(available_layers)} layers from alignment model"
                        )
                    else:
                        print(
                            f"Warning: No alignment layers found in {embed_proj_path}"
                        )
                else:
                    # This is a regular nn.Linear state dict - load into all embed_projs
                    for layer, embed_proj in enumerate(base_model.embed_projs):
                        embed_proj.load_state_dict(embed_proj_weights)
                        print(f"Loaded regular embed_proj weights into layer {layer}")

                # Don't train embed_proj
                for param in base_model.embed_projs.parameters():
                    param.requires_grad = False
            else:
                print(
                    "Warning: Model does not have embed_projs layer, skipping embed_proj loading"
                )
        else:
            print(f"Warning: embed_proj_path {embed_proj_path} does not exist")

    return base_model, target_model, model_type


def make_model(
    config: Dict,
    model_special_tokens_ids: Optional[Dict[str, int]] = None,
    output_dir: str = "",
    train_self: bool = False,
    use_embed_proj: bool = False,
    embed_proj_path: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedModel]:
    """
    Initialize and configure the training model and target model based on config.

    Args:
        config: Configuration dictionary containing model settings
        model_special_tokens_ids: Dictionary mapping special token names to IDs
        output_dir: Output directory for saving reserved token embeddings
        train_self: Whether training the model on itself

    Returns:
        Tuple of (wrapped_model, target_model)
    """
    use_peft_lora = config["train"].get(
        "peft_lora", True
    )  # Default to LoRA if not specified

    # Get embed_proj_path from config if not provided
    if embed_proj_path is None:
        embed_proj_path = config.get("embed_proj_path", None)

    base_model, target_model, model_type = load_models(
        predictor_model_path=config["model_path"],
        target_model_path=config.get("target_model_path", None),
        special_tokens_ids=model_special_tokens_ids,
        cache_dir=config.get("cache_dir", None),
        train_self=train_self,
        use_bf16=config["train"].get("bf16", True),
        batch_size=config["test"]["batch_size"],
        ckpt_dir=output_dir,
        use_embed_proj=use_embed_proj,
        embed_proj_path=embed_proj_path,
    )

    if use_peft_lora:
        # Use LoRA fine-tuning
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]

        # Add embed_projs to target modules if they exist
        if hasattr(base_model, "embed_projs") and base_model.embed_projs is not None:
            # Add each embed_proj layer to target modules
            for i in range(len(base_model.embed_projs)):
                if base_model.embed_projs[i].requires_grad:
                    target_modules.append(f"embed_projs.{i}")

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=config["train"].get("lora_r", 64),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        # make peft model
        wrapped_model = get_peft_model(
            base_model,
            peft_config,
            task_type=model_type,
            special_tokens_ids=model_special_tokens_ids,
            subject_embed_dim=(
                None if target_model is None else target_model.config.hidden_size
            ),
        )
    else:
        # Use full fine-tuning
        print("Using full fine-tuning (all parameters will be updated)")
        wrapped_model = base_model

    # Initialize target model (the model being interpreted)
    if train_self:
        target_model = wrapped_model
    elif not train_self and "target_model_path" not in config:
        # Same architecture (NOTE: may not be same model)
        target_model = base_model

    target_model.eval()

    return wrapped_model, target_model


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    task_type: Optional[str] = None,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
    special_tokens_ids: Optional[Dict[str, int]] = None,
    subject_embed_dim: int | None = None,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        task_type (`str`, `optional`, defaults to `None`):
            The type of task to be used for the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as
            False if you intend on training the model, unless the adapter weights will be replaced by different weights
            before training starts.
    """
    model_config = BaseTuner.get_model_config(model)
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    if mixed:
        # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    if task_type is None:
        task_type = peft_config.task_type

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return ContinuousPeft(
        model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        special_tokens_ids=special_tokens_ids,
        subject_embed_dim=subject_embed_dim,
    )
