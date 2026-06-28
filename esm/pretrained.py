from typing import Callable, Optional
from pathlib import Path
import json

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.models.function_decoder import FunctionTokenDecoder
from esm.layers.rotary import RotaryEmbedding
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_300M,
    ESMC_600M,
    ESMC_6B,
)

ModelBuilder = Callable[[torch.device | str], nn.Module]


def _load_sharded_safetensors(snapshot_path: Path, device) -> dict:
    from safetensors.torch import load_file
    with open(snapshot_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard in shard_files:
        state_dict.update(load_file(snapshot_path / shard, device=str(device)))
    return state_dict


def _remap_biohub_state_dict(state_dict: dict) -> dict:
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("esmc."):
            k = k[len("esmc."):]
        elif k.startswith("lm_head."):
            k = "sequence_head." + k[len("lm_head."):]
        remapped[k] = v
    return remapped


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESMC_300M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True, init_contact_head: bool = False, contact_head_weights_path: Optional[str] = None):
    with torch.device(device):
        print(init_contact_head)
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
            init_contact_head=init_contact_head,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth",
        map_location=device,
    )
    if init_contact_head:
        assert contact_head_weights_path is not None, "contact_head_weights_path must be provided if init_contact_head is True"
        contact_head_state_dict = torch.load(contact_head_weights_path, map_location=device)
        state_dict['contact_head.regression.weight'] = contact_head_state_dict['weight']
        state_dict['contact_head.regression.bias'] = contact_head_state_dict['bias']
    model.load_state_dict(state_dict)

    return model


def ESMC_600M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True, init_contact_head: bool = False, contact_head_weights_path: Optional[str] = None):
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
            init_contact_head=init_contact_head,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    if init_contact_head:
        assert contact_head_weights_path is not None, "contact_head_weights_path must be provided if init_contact_head is True"
        contact_head_state_dict = torch.load(contact_head_weights_path, map_location=device)
        state_dict['contact_head.regression.weight'] = contact_head_state_dict['weight']
        state_dict['contact_head.regression.bias'] = contact_head_state_dict['bias']
    model.load_state_dict(state_dict)

    return model


def ESMC_6B_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True, init_contact_head: bool = False, contact_head_weights_path: Optional[str] = None):
    # Build the skeleton on the meta device so the 6B random-init weights cost ~0 bytes,
    # then load the shards straight onto `device` and assign them in place. This avoids
    # ever holding a full copy of the model in CPU RAM (the old CPU path peaked at ~50 GB
    # and would OOM/swap-thrash an interactive job's memory cgroup).
    with torch.device("meta"):
        model = ESMC(
            d_model=2560,
            n_heads=40,
            n_layers=80,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
            init_contact_head=init_contact_head,
        ).eval()
    state_dict = _remap_biohub_state_dict(
        _load_sharded_safetensors(data_root("esmc-6b"), device)
    )
    if init_contact_head:
        assert contact_head_weights_path is not None, "contact_head_weights_path must be provided if init_contact_head is True"
        contact_head_state_dict = torch.load(contact_head_weights_path, map_location=device)
        state_dict['contact_head.regression.weight'] = contact_head_state_dict['weight']
        state_dict['contact_head.regression.bias'] = contact_head_state_dict['bias']
    # assign=True swaps the loaded tensors in directly instead of copying into preallocated
    # storage, so meta params are replaced rather than requiring a second allocation.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    assert not unexpected, f"Unexpected keys in state dict: {unexpected}"
    # The only tensors not present in the checkpoint are the non-persistent rotary buffers
    # (RotaryEmbedding.inv_freq). After a meta init + assign these are still on `meta`, so
    # rematerialize them on the target device.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.device = device
            module.reset_parameters()
    remaining_meta = [n for n, p in model.named_parameters() if p.is_meta]
    remaining_meta += [n for n, b in model.named_buffers() if b.is_meta]
    assert not remaining_meta, f"Tensors left on meta device after load: {remaining_meta}"
    return model


def ESM3_sm_open_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESMC_600M: ESMC_600M_202412,
    ESMC_300M: ESMC_300M_202412,
    ESMC_6B: ESMC_6B_202412,
}


def load_local_model(
    model_name: str, device: torch.device = torch.device("cpu"), use_flash_attn: bool = True, 
    init_contact_head: bool = False, contact_head_weights_path: Optional[Path] = None,
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    params = {"device": device}
    if model_name in [ESMC_300M, ESMC_600M, ESMC_6B]:
        params["use_flash_attn"] = use_flash_attn
        params["init_contact_head"] = init_contact_head
        params["contact_head_weights_path"] = contact_head_weights_path
    return LOCAL_MODEL_REGISTRY[model_name](**params)


# Register custom versions of ESM3 for use with the local inference API
def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
