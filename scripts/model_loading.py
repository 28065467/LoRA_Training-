"""Shared helper so base_eval.py / reload_and_eval.py can load either a plain text CausalLM
(Qwen3-Coder-30B-A3B-Instruct, ...) or a VL-wrapped checkpoint whose HF architecture is only
registered under AutoModelForImageTextToText (e.g. Qwen3.8-27B's `Qwen3_5ForConditionalGeneration`
— it declares `language_model_only: false` even though we only ever feed it text).
AutoModelForCausalLM.from_pretrained would raise `Unrecognized configuration class` for the
latter, so pick the right Auto class up front from the config instead of trying and catching.
"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def load_causal_model(model_id_or_path, **kwargs):
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    architectures = getattr(config, "architectures", None) or []
    if any(a in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values() for a in architectures):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id_or_path, config=config, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id_or_path, config=config, **kwargs)


def load_tokenizer(model_id_or_path, **kwargs):
    return AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True, **kwargs)
