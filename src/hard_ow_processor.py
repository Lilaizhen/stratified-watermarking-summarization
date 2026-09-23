"""Hard original watermark: ban red tokens, preserving the forced BOS step."""
import torch
from transformers import LogitsProcessor, WatermarkLogitsProcessor


class HardOWProcessor(LogitsProcessor):
    def __init__(self, vocab_size, device, gamma, key, forced_bos_token_id):
        self.colors = WatermarkLogitsProcessor(
            vocab_size, device, greenlist_ratio=gamma, bias=0,
            hashing_key=key, seeding_scheme='lefthash', context_width=1)
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids, scores):
        # BART's existing forced-BOS processor precedes custom processors.
        # This structural token is excluded by the detector, not a content fallback.
        if input_ids.shape[-1] == 1 and self.forced_bos_token_id is not None:
            return scores
        green = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(len(scores)):
            green[i, self.colors._get_greenlist_ids(input_ids[i])] = True
        result = scores.masked_fill(~green, -torch.inf)
        if not torch.isfinite(result).any(-1).all():
            raise RuntimeError('Hard OW has no finite green candidate; no red fallback is permitted')
        return result
