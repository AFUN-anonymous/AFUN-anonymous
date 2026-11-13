"""Construct the same model for training and evaluation."""
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


def build_model(cfg, device="cuda", *, alignment=False, evaluation=False, legacy_afun_pt=False):
    """Build AFUN. `legacy_afun_pt` restores the three preprocessing quirks that legacy checkpoints
    (afun.pt among them) expect: Qwen3-VL sees a near-black
    image, the MetaQuery RMSNorm stays bf16 after Stage 1, and depth is resized bilinearly.
    load_model sets it from the checkpoint; training always uses the fixed preprocessing."""
    from .alignment import AlignmentModel
    from .end_to_end import EndToEndModel

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.llm.device = str(device)
        cfg.llm.llm_cfg.legacy_afun_pt = legacy_afun_pt
        cfg.decoder.model_cfg.device = str(device)
        if evaluation:
            cfg.decoder.training = False
            cfg.decoder.loss = None
    llm = instantiate(cfg.llm)
    if str(device).startswith("cuda"):
        import torch
        llm.to(torch.bfloat16)
        if llm.rms_norm_layer is not None and (alignment or not legacy_afun_pt):
            # The MetaQuery RMSNorm trains in fp32: in bf16, Adam's ~2e-4 updates are below the spacing
            # near 1.0 (0.0078) and round away. afun.pt kept it in bf16 after Stage 1.
            llm.rms_norm_layer.float()
    cfg.decoder.model_cfg.projector_in_dim = llm.llm_hidden_size
    if cfg.decoder.get("motion_processor") is not None:
        cfg.decoder.motion_processor.model_cfg.kv_dim = llm.llm_hidden_size
    decoder = instantiate(cfg.decoder, _recursive_=False)
    model_class = AlignmentModel if alignment else EndToEndModel
    return model_class(encoder=None, llm=llm, decoder=decoder, rms_norm=cfg.get("rms_norm", False),
                       legacy_afun_pt=legacy_afun_pt)
