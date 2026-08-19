from collections.abc import Mapping, Sequence
import torch
import torch.distributed as dist

def iter_learnable_tensors(tree, prefix="root"):
    """Yield leaf te nsors with requires_grad=True from nested dict/list/tuple, 
    and print non-leaf tensor info."""
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            yield from iter_learnable_tensors(v, prefix=f"{prefix}.{k}")
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for i, v in enumerate(tree):
            yield from iter_learnable_tensors(v, prefix=f"{prefix}[{i}]")
    elif torch.is_tensor(tree):
        if tree.requires_grad:
            if tree.is_leaf:
                yield tree
            else:
                print(f"⚠️ Non-leaf tensor at path '{prefix}': "
                    f"shape={tuple(tree.shape)}, "
                    f"grad_fn={tree.grad_fn}")
                # optionally still yield it or raise
                raise ValueError(f"Found non-leaf tensor at '{prefix}'")


@torch.no_grad()
def broadcast_loradict(loradict: dict, src: int = 0) -> int:
    """Broadcast learnable LoRA tensors from ``src`` to every DDP rank.
    """
    if loradict is None or not (dist.is_available() and dist.is_initialized()):
        return 0

    tensors = list(iter_learnable_tensors(loradict))
    for tensor in tensors:
        dist.broadcast(tensor, src=src)
    return len(tensors)


@torch.no_grad()
def average_loradict_gradients(loradict: dict) -> int:
    """Average external LoRA gradients across all initialized DDP ranks.

    Returns the number of gradients averaged. This is a no-op for a
    single-process run.
    """
    if loradict is None or not (dist.is_available() and dist.is_initialized()):
        return 0

    tensors = list(iter_learnable_tensors(loradict))
    if not tensors:
        return 0

    world_size = dist.get_world_size()
    grad_counts = torch.tensor(
        [int(tensor.grad is not None) for tensor in tensors],
        dtype=torch.int32,
        device=tensors[0].device,
    )
    dist.all_reduce(grad_counts, op=dist.ReduceOp.SUM)

    inconsistent = (grad_counts != 0) & (grad_counts != world_size)
    if bool(inconsistent.any().item()):
        indices = inconsistent.nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "LoRA gradient presence differs across DDP ranks for tensor "
            f"indices {indices[:20]}"
        )

    synced = 0
    for tensor, count in zip(tensors, grad_counts.tolist()):
        if count == 0:
            continue
        dist.all_reduce(tensor.grad, op=dist.ReduceOp.SUM)
        tensor.grad.div_(world_size)
        synced += 1

    return synced

def merge_loradicts(lora1: dict, lora2: dict, method: str) -> dict:
    """
    Merge two loradicts by concatenating along the LoRA rank dimension r.

    Leaf tensors:
      A: [Lb, in, r]   -> concat dim=2  => [Lb, in, r1+r2]
      B: [Lb, r, out]  -> concat dim=1  => [Lb, r1+r2, out]
      C: [Lb, out]     -> sum           => [Lb, out]

    Assumes lora1 and lora2 have identical key structure and same Lb/in/out.
    """
    
    assert method == "rl", f"Unsupported merge method: {method}, only support rl for now."
    if lora1 is None:
        return lora2
    if lora2 is None:
        return lora1
    
    def _merge_leaf(d1, d2, path=""):
        # Leaf node: {"A":..., "B":..., "C":...}
        if isinstance(d1, dict) and "A" in d1 and "B" in d1:
            A1, B1 = d1["A"], d1["B"]
            A2, B2 = d2["A"], d2["B"]

            if A1 is None or A2 is None or B1 is None or B2 is None:
                raise ValueError(f"{path}: A/B cannot be None for rank-concat merge.")

            # Check batch + core dims match
            if A1.shape[0] != A2.shape[0]:
                raise ValueError(f"{path}.A: Lb mismatch {A1.shape[0]} vs {A2.shape[0]}")
            if B1.shape[0] != B2.shape[0]:
                raise ValueError(f"{path}.B: Lb mismatch {B1.shape[0]} vs {B2.shape[0]}")

            # A: [Lb, in, r]
            if A1.shape[1] != A2.shape[1]:
                raise ValueError(f"{path}.A: in_features mismatch {A1.shape[1]} vs {A2.shape[1]}")
            # B: [Lb, r, out]
            if B1.shape[2] != B2.shape[2]:
                raise ValueError(f"{path}.B: out_features mismatch {B1.shape[2]} vs {B2.shape[2]}")

            # Consistency: A.in must match B.r matmul expectation via tmp
            # (We don't need A.r == B.r here because we are concatenating them,
            #  but each pair must be internally consistent.)
            if A1.shape[2] != B1.shape[1]:
                raise ValueError(f"{path}: r mismatch inside lora1: A.r={A1.shape[2]} vs B.r={B1.shape[1]}")
            if A2.shape[2] != B2.shape[1]:
                raise ValueError(f"{path}: r mismatch inside lora2: A.r={A2.shape[2]} vs B.r={B2.shape[1]}")

            out = {
                "A": torch.cat([A1, A2], dim=2),  # concat r
                "B": torch.cat([B1, B2], dim=1),  # concat r
            }

            C1 = d1.get("C", None)
            C2 = d2.get("C", None)
            if (C1 is None) != (C2 is None):
                raise ValueError(f"{path}.C: one is None, the other is not.")
            out["C"] = None if C1 is None else (C1 + C2)

            return out

        # Recurse through nested dicts
        if isinstance(d1, dict) and isinstance(d2, dict):
            if d1.keys() != d2.keys():
                raise ValueError(f"{path}: key mismatch {set(d1.keys())} vs {set(d2.keys())}")
            return {
                k: _merge_leaf(
                    d1[k],
                    d2[k],
                    path=f"{path}.{k}" if path else str(k),
                )
                for k in d1.keys()
            }

        raise TypeError(f"{path}: unsupported types {type(d1)} and {type(d2)}")

    return _merge_leaf(lora1, lora2)

def freeze_loradict(loradict: dict) -> dict:
    """
    Freeze all torch.Tensors inside a nested loradict IN-PLACE
    by setting requires_grad_(False).

    - Does NOT detach
    - Does NOT replace tensor objects
    - Safe for shared references / caching

    Returns the same loradict object (mutated).
    """

    def _walk(obj):
        if torch.is_tensor(obj):
            obj.requires_grad_(False)   # <-- ONLY this
            return

        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return

        if isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
            return

        # ignore None / scalars
        return

    _walk(loradict)
    return loradict

def loradict_all_requires_grad(loradict: dict, expected: bool, *, verbose: bool = True) -> bool:
    """
    Check whether all torch.Tensors inside a nested loradict
    have requires_grad == expected, and print all mismatches.

    Args:
        loradict: nested dict structure containing torch.Tensors
        expected: True (all trainable) or False (all frozen)
        verbose: if True, print all mismatched tensors

    Returns:
        bool: True if all tensors match, otherwise False
    """
    if loradict is None:
        return True

    wrong = []  # collect (path, tensor) for mismatches

    def _walk(obj, path="root"):
        if torch.is_tensor(obj):
            if obj.requires_grad != expected:
                wrong.append((path, obj))
            return obj.requires_grad == expected

        if isinstance(obj, dict):
            ok = True
            for k, v in obj.items():
                ok = _walk(v, f"{path}.{k}") and ok
            return ok

        if isinstance(obj, (list, tuple)):
            ok = True
            for i, v in enumerate(obj):
                ok = _walk(v, f"{path}[{i}]") and ok
            return ok

        return True  # ignore None / non-tensors

    all_ok = _walk(loradict)

    if verbose and wrong:
        print(f"[loradict_all_requires_grad] Found {len(wrong)} tensor(s) with requires_grad != {expected}:")
        for p, t in wrong:
            print(
                f" - {p}: requires_grad={t.requires_grad}, "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"
            )

    return all_ok
