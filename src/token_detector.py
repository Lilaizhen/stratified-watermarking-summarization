import math
import torch

def score_tokens(ids, processor, special_ids):
    # Keep original adjacency: deleting an interior special token changes the seed.
    seen = set()
    green = total = 0
    for prev, current in zip(ids[:-1], ids[1:]):
        pair = (prev, current)
        if prev in special_ids or current in special_ids or pair in seen:
            continue
        seen.add(pair)
        prefix = torch.tensor([prev], device=processor.rng.device)
        green_ids = processor._get_greenlist_ids(prefix)
        green += int((green_ids == current).any().item())
        total += 1
    gamma = processor.greenlist_size / processor.vocab_size
    z = (green - gamma * total) / math.sqrt(total * gamma * (1 - gamma)) if total else None
    return {"tokens_scored": total, "green_tokens": green, "z": z,
            "nominal_p": math.erfc(z / math.sqrt(2)) / 2 if z is not None else None}

