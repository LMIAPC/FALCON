# models/encoders/__init__.py
from .base import _ENCODER_REGISTRY
from . import dino, clip, resnet, bmclip  # noqa

def build_encoder(cfg_module):
    """
    Construct an encoder instance based on configuration.
    
    Args:
        cfg_module: The complete config.py module object containing:
            - ENCODER: Encoder-specific configuration
            - MODEL_CONFIG: Model architecture configuration
            - PATHS: Path configuration
    
    Returns:
        An initialized encoder instance from the registry
    
    Raises:
        ValueError: If specified encoder type is not found in registry
    """
    enc_cfg = cfg_module.ENCODER
    model_cfg = cfg_module.MODEL_CONFIG

    encoder_type = enc_cfg['type']
    if encoder_type not in _ENCODER_REGISTRY:
        available_encoders = list(_ENCODER_REGISTRY.keys())
        raise ValueError(f"[build_encoder] Unknown encoder type '{encoder_type}'. "
                        f"Available encoders: {available_encoders}")

    # Pass the complete cfg_module to let encoder classes access:
    # - MODEL_CONFIG for architecture parameters
    # - PATHS for weight locations
    return _ENCODER_REGISTRY[encoder_type].build(
        encoder_cfg=enc_cfg,
        model_cfg=model_cfg,
        paths=cfg_module.PATHS
    )