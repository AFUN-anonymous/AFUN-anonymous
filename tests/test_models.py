from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from src.models import build
from src.models.llm import qwen_wrapper

QWEN = "Qwen/Qwen3-VL-8B-Instruct"


class _NoWeights(nn.Module):
    """Stands in for the 8B Qwen3-VL weights, which tokenize does not use."""
    config = SimpleNamespace(hidden_size=8)

    @property
    def device(self):
        return torch.device("cpu")


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(QWEN, local_files_only=True)
    except OSError:
        pytest.skip(f"the {QWEN} processor is not in the Hugging Face cache")


@pytest.fixture
def qwen(processor, monkeypatch):
    """QwenWrapper with the real Qwen3-VL processor and no model weights."""
    monkeypatch.setattr(qwen_wrapper.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(qwen_wrapper.AutoModelForImageTextToText, "from_pretrained",
                        lambda *args, **kwargs: _NoWeights())
    return lambda **llm_cfg: qwen_wrapper.QwenWrapper(
        model_name=QWEN, llm_cfg={"rms_norm": False, **llm_cfg}, device="cpu")


def _test_card():
    """A 3 x 48 x 64 uint8 image with a different gradient in each channel."""
    y, x = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
    return torch.stack([x * 4, y * 5, (x + y) * 2]).to(torch.uint8)


def test_qwen_sees_the_image_it_is_given_as_floats(qwen, processor):
    card = _test_card()
    reference = processor.image_processor(
        images=[Image.fromarray(card.permute(1, 2, 0).numpy())], return_tensors="pt")["pixel_values"]
    tokens = qwen().tokenize(["What is in the image?"], image=[card.float() / 255])
    torch.testing.assert_close(tokens["pixel_values"][0], reference)


def test_legacy_afun_pt_keeps_the_near_black_image(qwen, processor):
    black = processor.image_processor(images=[Image.new("RGB", (64, 48))], return_tensors="pt")["pixel_values"]
    tokens = qwen(legacy_afun_pt=True).tokenize(["What is in the image?"], image=[_test_card().float() / 255])
    # The processor divides the [0, 1] floats by 255 again: afun.pt's Qwen saw pixels within 1/255 of black.
    bound = 1.02 * (1 / 255) / min(processor.image_processor.image_std)
    assert (tokens["pixel_values"][0] - black).abs().max() <= bound


def test_qwen_rejects_float_images_outside_unit_range(qwen):
    sam3_normalized = _test_card().float() / 127.5 - 1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        qwen().tokenize(["What is in the image?"], image=[sam3_normalized])


class _TinyLLM(nn.Module):
    llm_hidden_size = 4

    def __init__(self, llm_cfg):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.llm_model = nn.Linear(4, 4)
        self.rms_norm_layer = nn.RMSNorm((2, 4))


def _build(monkeypatch, **kwargs):
    """build_model with Qwen3-VL and SAM3 replaced by tiny modules; its dtype handling stays real."""
    monkeypatch.setattr(build, "instantiate",
                        lambda node, **_: _TinyLLM(node.llm_cfg) if "llm_cfg" in node else nn.Identity())
    cfg = OmegaConf.create({"llm": {"_target_": "qwen", "llm_cfg": {}},
                            "decoder": {"_target_": "sam3", "model_cfg": {}}})
    # A "cuda" device selects the bf16 cast; the tiny modules themselves stay on the CPU.
    return build.build_model(cfg, "cuda", **kwargs)


@pytest.mark.parametrize("stage1", [True, False])
def test_metaquery_rms_norm_trains_in_fp32_in_every_stage(monkeypatch, stage1):
    model = _build(monkeypatch, alignment=stage1)
    assert model.llm.rms_norm_layer.weight.dtype == torch.float32
    assert model.llm.llm_model.weight.dtype == torch.bfloat16


def test_legacy_afun_pt_reaches_all_three_quirks(monkeypatch):
    model = _build(monkeypatch, evaluation=True, legacy_afun_pt=True)
    assert model.llm.rms_norm_layer.weight.dtype == torch.bfloat16
    assert model.llm.llm_cfg.legacy_afun_pt  # read by QwenWrapper.tokenize
    assert model.legacy_afun_pt              # read by inference for the depth resize
