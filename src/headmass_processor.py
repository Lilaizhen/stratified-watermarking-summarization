"""Regional green tilt. The final protocol passes special_ids=[]: no extra special-token protection."""
import math
import torch
from transformers import LogitsProcessor, WatermarkLogitsProcessor


def partition(scores, special_ids, rho):
    ordinary = torch.ones_like(scores, dtype=torch.bool)
    ordinary[:, special_ids] = False
    if rho is None:
        return [ordinary]
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)
    p = scores.softmax(-1).gather(1, ids)
    head = torch.zeros_like(ordinary).scatter(1, ids, p.cumsum(-1)-p < rho) & ordinary
    return [head, ordinary & ~head]


def mass_preserving_tilt(scores, green, special_ids, rho, delta):
    original = scores.float()
    if delta == 0:
        return original.clone()
    out = original.clone()
    tilted = original + delta * green.float()
    for bucket in partition(original, special_ids, rho):
        before = original.masked_fill(~bucket, -torch.inf).logsumexp(-1)
        after = tilted.masked_fill(~bucket, -torch.inf).logsumexp(-1)
        active = torch.isfinite(before)
        correction = torch.where(active, before, 0) - torch.where(active, after, 0)
        out = torch.where(bucket & active[:, None], tilted + correction[:, None], out)
    return out


class HeadMassProcessor(LogitsProcessor):
    def __init__(self, vocab_size, device, gamma, key, delta, special_ids, rho=None):
        if not math.isfinite(delta) or delta < 0 or (rho is not None and not 0 < rho < 1):
            raise ValueError('Invalid headmass parameters')
        self.colors = WatermarkLogitsProcessor(vocab_size, device, greenlist_ratio=gamma,
            bias=0, hashing_key=key, seeding_scheme='lefthash', context_width=1)
        self.special_ids = [i for i in special_ids if 0 <= i < vocab_size]
        self.rho, self.delta = rho, delta

    def __call__(self, input_ids, scores):
        green = torch.zeros_like(scores, dtype=torch.bool)
        for b in range(len(scores)):
            green[b, self.colors._get_greenlist_ids(input_ids[b])] = True
        return mass_preserving_tilt(scores, green, self.special_ids, self.rho, self.delta)
