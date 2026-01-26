import os, json, torch
from PIL import Image
from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS
from .base import BaseEncoder, register_encoder
from .utils import split_sliding_patches

@register_encoder("bmclip_b16")
class FrozenBMCLIP(BaseEncoder):
    def __init__(self, local_path: str, patch_window: tuple[int, int], patch_stride: tuple[int, int], feature_dim: int):
        super().__init__()

        model_bin = os.path.join(local_path, "open_clip_pytorch_model.bin")
        cfg_json = os.path.join(local_path, "open_clip_config.json")
        assert os.path.isfile(model_bin), f"missing {model_bin}"
        assert os.path.isfile(cfg_json), f"missing {cfg_json}"

        with open(cfg_json, "r") as f:
            cfg_loaded = json.load(f)
        model_cfg = cfg_loaded["model_cfg"]
        preprocess_cfg = cfg_loaded["preprocess_cfg"]

        # Dummy text config (ensures we don't need to download text branch)
        model_cfg["text_cfg"] = {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12,
        }
        local_name = "biomedclip_local_" + os.path.basename(local_path)
        if local_name not in _MODEL_CONFIGS:
            _MODEL_CONFIGS[local_name] = model_cfg

        # Create model architecture (without loading weights)
        model, _, preprocess = create_model_and_transforms(
            model_name=local_name,
            pretrained="",  # Don't auto-load weights
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        # Manually load vision branch parameters with strict=False
        state_dict = torch.load(model_bin, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remove 'module.' prefix (compatibility for distributed/single-GPU storage)
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                     for k, v in state_dict.items()}
        # Strictly keep only vision branch parameters
        visual_keys = [k for k in state_dict if k.startswith("visual.")]
        visual_state_dict = {k: state_dict[k] for k in visual_keys}
        missing, unexpected = model.visual.load_state_dict(visual_state_dict, strict=False)
        if missing or unexpected:
            print("[BMCLIP] Non-critical missing/unexpected keys:", missing, unexpected)

        self.visual = model.visual
        self.preprocess = preprocess
        self.patch_window = patch_window
        self.patch_stride = patch_stride
        self.feature_dim = feature_dim
        self.visual.eval().requires_grad_(False)

    @classmethod
    def build(cls, *, encoder_cfg, model_cfg, paths):
        typ = encoder_cfg["type"]
        loc = paths["pretrained_model"](typ)
        dim = encoder_cfg["dims"][typ]
        patch_window = tuple(model_cfg.get("patch_window", (224, 224)))
        patch_stride = tuple(model_cfg.get("patch_stride", (224, 224)))
        return cls(loc, patch_window, patch_stride, dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = (x * 0.5 + 0.5).clamp(0, 1)
        patches = split_sliding_patches(x_float, self.patch_window, self.patch_stride)
        flat = torch.cat(patches, 0)
        N = len(patches)

        # Must convert to PIL images
        pil_imgs = [Image.fromarray((img.permute(1,2,0).cpu().numpy() * 255).astype('uint8')) 
                   for img in flat]
        batch = torch.stack([self.preprocess(img) for img in pil_imgs]).to(x.device)  # (B*N, 3, 224, 224)

        feats = self.visual(batch)   # (B*N, D)
        return feats.view(N, B, self.feature_dim).permute(1, 0, 2)
