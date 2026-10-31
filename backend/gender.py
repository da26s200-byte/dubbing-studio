import json
from pathlib import Path

import numpy as np

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "wav2vec2-age-gender"

MAX_SECONDS = 30.0  # speech per speaker the classifier hears (their longest turns first)
WINDOW = 10.0  # longest audio in one forward pass
MIN_TURN = 1.0  # turns / window remainders shorter than this are skipped
MIN_SPEECH = 3.0  # a speaker with less usable speech than this is "unknown"
MIN_CONF = 0.65  # the winning side of female vs male must reach this, else "unknown"

_model = None


def get_model():
    """(processor, model, device); loaded on first use and reused. Needs `transformers` and the weights in
    MODEL_DIR (see MODEL_NAME on Hugging Face). The model card's custom class is reproduced here."""
    global _model
    if _model is None:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                x = torch.tanh(self.dense(self.dropout(x)))
                return self.out_proj(self.dropout(x))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)  # female, male, child
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return torch.softmax(self.gender(hidden), dim=1)

        if not MODEL_DIR.exists():
            raise RuntimeError(f"gender model not found in {MODEL_DIR}; download {MODEL_NAME} there")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
        model = AgeGenderModel.from_pretrained(str(MODEL_DIR)).to(device).eval()
        _model = (processor, model, device)
    return _model


def _windows(data, sr, turns):
    """The audio the classifier hears for one speaker: their longest turns, cut into windows of at most
    WINDOW seconds, MAX_SECONDS in all. [(samples, seconds)]"""
    out, left = [], MAX_SECONDS
    for start, end in sorted(turns, key=lambda t: t[0] - t[1]):
        if left <= 0:
            break
        if end - start < MIN_TURN:
            continue
        end = min(end, start + left)
        t = start
        while t < end:
            b = min(t + WINDOW, end)
            if b - t >= MIN_TURN:
                out.append((data[int(t * sr) : int(b * sr)], b - t))
            t = b
        left -= end - start
    return out


def classify_speakers(data, sr: int, turns) -> dict:
    """Gender per speaker from their speech in `data` (mono float32 at 16 kHz), `turns` being
    [(speaker, start, end)]. Returns {speaker: {"gender": "female" | "male" | "unknown", "confidence",
    "seconds"}}: female vs male probability (the child class is ignored) averaged over the speaker's windows,
    weighted by length."""
    import torch

    processor, model, device = get_model()
    by_speaker = {}
    for speaker, a, b in turns:
        by_speaker.setdefault(speaker, []).append((a, b))

    out = {}
    for speaker in sorted(by_speaker):
        windows = _windows(data, sr, by_speaker[speaker])
        seconds = sum(s for _, s in windows)
        p_female = 0.0
        for samples, s in windows:
            x = processor(samples, sampling_rate=sr)["input_values"][0].reshape(1, -1)
            with torch.inference_mode():
                female, male, _ = model(torch.from_numpy(x).to(device))[0].tolist()
            p_female += s * female / max(female + male, 1e-9)
        p_female = p_female / seconds if seconds else 0.5
        conf = max(p_female, 1 - p_female)
        gender = "unknown"
        if seconds >= MIN_SPEECH and conf >= MIN_CONF:
            gender = "female" if p_female >= 0.5 else "male"
        out[speaker] = {"gender": gender, "confidence": round(conf, 3), "seconds": round(seconds, 1)}
    return out


def load_details(jdir: Path) -> dict:
    """The per-speaker results ({speaker: {"gender", "confidence", "seconds"}}) saved in a run's
    diarization.json; {} for runs from before gender detection."""
    try:
        genders = json.loads((jdir / "diarization.json").read_text()).get("genders") or {}
    except (OSError, ValueError):
        return {}
    return {int(s): g for s, g in genders.items()}


def load(jdir: Path) -> dict:
    """{speaker: "female" | "male"} of a run; speakers of unknown gender are left out."""
    return {s: g["gender"] for s, g in load_details(jdir).items() if g["gender"] in ("female", "male")}
