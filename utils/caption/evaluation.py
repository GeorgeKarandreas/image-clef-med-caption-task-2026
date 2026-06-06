"""Evaluation utilities for caption task."""

import re
import string

import evaluate
import numpy as np
from tqdm.auto import tqdm
import torch

from bert_score import BERTScorer
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()

    total_loss = 0.0

    for batch in tqdm(loader):
        biomed_pixels = batch["biomed_pixels"].to(device, non_blocking=True)
        swin_pixels = batch["swin_pixels"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                biomed_pixels=biomed_pixels,
                swin_pixels=swin_pixels,
                labels=labels,
            )

        total_loss += outputs.loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_captions(
    model,
    loader,
    device,
    max_new_tokens=64,
    num_beams=3,
):
    model.eval()

    predictions = []
    references = []

    for batch in tqdm(loader):
        biomed_pixels = batch["biomed_pixels"].to(device, non_blocking=True)
        swin_pixels = batch["swin_pixels"].to(device, non_blocking=True)

        generated = model.generate_caption(
            biomed_pixels=biomed_pixels,
            swin_pixels=swin_pixels,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        predictions.extend(generated)
        references.extend(batch["caption"])

    return evaluation_metrics(
        predictions=predictions,
        references=references,
        device=device.type if hasattr(device, "type") else str(device),
    )


def evaluation_metrics(
    predictions,
    references,
    device: str | None = None,
    text_batch_size: int = 8,
):
    """
    Computes competition caption metrics.

    Returns aggregate mean scores for:
    - BERTScore F1 using ``microsoft/deberta-xlarge-mnli``
    - ROUGE-1 F1 with no stemmer
    - BLEURT using ``lucadiliello/BLEURT-20-D12``

    ``predictions`` and ``references`` can each be either:
    - matching dicts keyed by image id
    - matching sequences of captions
    """
    rouge_scorer = evaluate.load("rouge")

    def _preprocess_caption(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r"\d+", "number", text)
        return text.translate(str.maketrans("", "", string.punctuation))

    def _normalize_pairs(preds, refs):
        if isinstance(preds, dict) and isinstance(refs, dict):
            pred_keys = list(preds.keys())
            ref_keys = set(refs.keys())
            missing_keys = [key for key in pred_keys if key not in ref_keys]
            if missing_keys:
                raise ValueError(
                    "references is missing prediction keys: "
                    + ", ".join(map(str, missing_keys[:10]))
                )
            return [
                (_preprocess_caption(preds[key]), _preprocess_caption(refs[key]))
                for key in pred_keys
            ]

        if isinstance(preds, (list, tuple)) and isinstance(refs, (list, tuple)):
            if len(preds) != len(refs):
                raise ValueError(
                    "predictions and references must have the same length."
                )
            return [
                (_preprocess_caption(pred), _preprocess_caption(ref))
                for pred, ref in zip(preds, refs)
            ]

        raise TypeError(
            "predictions and references must both be dicts or both be sequences."
        )

    pairs = _normalize_pairs(predictions, references)

    if not pairs:
        raise ValueError("predictions and references must not be empty.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_scorer = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli",
        idf=True,
        idf_sents=[ref for _, ref in pairs],
        device=device,
    )

    bert_scores = []
    rouge_scores = []

    for pred, ref in pairs:
        if ref or pred:
            bert_f1 = bert_scorer.score(cands=[pred], refs=[ref])[2].item()
            rouge_f1 = rouge_scorer.compute(
                predictions=[pred],
                references=[ref],
                use_aggregator=False,
                use_stemmer=False,
            )["rouge1"]
        else:
            bert_f1 = 1.0
            rouge_f1 = 1.0

        bert_scores.append(bert_f1)
        rouge_scores.append(rouge_f1)

    if device == "cuda":
        bert_scorer = None
        torch.cuda.empty_cache()

    bleurt_config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20-D12")
    bleurt_model = BleurtForSequenceClassification.from_pretrained(
        "lucadiliello/BLEURT-20-D12",
        config=bleurt_config,
    )
    bleurt_tokenizer = BleurtTokenizer.from_pretrained(
        "lucadiliello/BLEURT-20-D12"
    )
    bleurt_model.to(device)
    bleurt_model.eval()

    bleurt_scores = []
    bs = max(1, int(text_batch_size))
    refs = [ref for _, ref in pairs]
    preds = [pred for pred, _ in pairs]

    for start in range(0, len(pairs), bs):
        batch_refs = refs[start : start + bs]
        batch_preds = preds[start : start + bs]

        with torch.inference_mode():
            inputs = bleurt_tokenizer(
                batch_refs,
                batch_preds,
                padding="longest",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_scores = bleurt_model(**inputs).logits.flatten().cpu().tolist()

        bleurt_scores.extend(batch_scores)

    if device == "cuda" and torch.cuda.is_available():
        bleurt_model = None
        bleurt_tokenizer = None
        bleurt_config = None
        torch.cuda.empty_cache()

    return {
        "bert": float(np.mean(bert_scores)),
        "rouge": float(np.mean(rouge_scores)),
        "bleurt": float(np.mean(bleurt_scores)),
    }
