import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import moderngl
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    BOS,
    EOS,
    PAD,
    DEVANAGARI,
    SANSKRIT_BOS,
    SANSKRIT_EOS,
    SANSKRIT_PAD,
    SANSKRIT_OFFSET,
    TwoStageGenerator,
    Tokenizer,
    config_args,
    paths,
    pick_device,
    resolve_artifact_path,
    sample_logits,
)


_DEVA_FONTS = [
    "Kohinoor Devanagari", "Devanagari Sangam MN", "Devanagari MT", "ITF Devanagari",
    "Lohit Devanagari", "Noto Sans Devanagari", "Sahadeva", "Samyak Devanagari",
]

LATIN_TEXT_FONTS = [
    "Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans",
]


def find_font(candidates):
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return available[name]
    return None


def find_fallback_font():
    for f in fm.fontManager.ttflist:
        if "DejaVu Sans" in f.name and "Bold" not in f.name and "Oblique" not in f.name:
            return f.fname
    return fm.fontManager.ttflist[0].fname if fm.fontManager.ttflist else None


def attach_capture(model):
    capture = {
        "stage1_self": [], "stage1_cross": [],
        "stage2_self": [], "stage2_cross": [],
    }
    handles = []

    def make_hooks(key, layer_idx):
        def pre_hook(module, args, kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs

        def fwd_hook(module, args, output):
            weights = output[1]
            if weights is not None:
                capture[key].append((layer_idx, weights.detach().cpu()))

        return pre_hook, fwd_hook

    def wire(attn_module, key, layer_idx):
        pre, fwd = make_hooks(key, layer_idx)
        handles.append(attn_module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(attn_module.register_forward_hook(fwd))

    for i, layer in enumerate(model.stage1.layers):
        wire(layer.self_attn, "stage1_self", i)
        wire(layer.multihead_attn, "stage1_cross", i)
    for i, layer in enumerate(model.stage2.layers):
        wire(layer.self_attn, "stage2_self", i)
        wire(layer.multihead_attn, "stage2_cross", i)

    def clear():
        for k in capture:
            capture[k].clear()

    def remove():
        for h in handles:
            h.remove()

    return capture, clear, remove


@torch.no_grad()
def run_stage1_frames(model, prompt_memory, prompt_pad, capture, clear_capture, args, device):
    ids = [SANSKRIT_BOS]
    running_chars = []
    frames = []
    states = []
    requested_limit = args.stage1_infer_chars or model.stage1_chars
    positional_capacity = int(model.stage1.pos.shape[1]) - 1
    model_limit = min(int(model.stage1_chars), positional_capacity)
    limit = min(int(requested_limit), model_limit)
    if int(requested_limit) > model_limit:
        print(
            f"warning: requested {requested_limit} Stage 1 characters, but this "
            f"checkpoint supports at most {model_limit}; limiting automatically",
            file=sys.stderr,
        )
    num_layers = len(model.stage1.layers)
    vocab_size = model.sanskrit_vocab

    for _ in range(limit):
        clear_capture()
        arr = torch.tensor([ids], dtype=torch.long, device=device)
        hidden, logits = model.stage1(ids=arr, memory=prompt_memory, memory_pad=prompt_pad)
        last_hidden = hidden[:, -1:, :]
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits / args.stage1_temperature, dim=-1).clone()
        probs[[SANSKRIT_PAD, SANSKRIT_BOS]] = 0.0
        probs = probs / probs.sum().clamp_min(1e-9)
        entropy = float(-(probs * (probs + 1e-9).log()).sum())
        idx = sample_logits(
            last_logits, args.stage1_temperature, [SANSKRIT_PAD, SANSKRIT_BOS],
            args.stage1_top_k, args.stage1_top_p,
        )

        self_layers = []
        for _, w in capture["stage1_self"][-min(3, num_layers):]:
            self_layers.extend(
                select_stable_head_maps(
                    attention_tensor_to_head_maps(w),
                    max_maps=max(1, int(getattr(args, "head_maps_per_tensor", 2))),
                )
            )
        states.append(last_hidden)
        hidden_vec = last_hidden[0, 0, :].cpu().numpy()

        is_eos = idx == SANSKRIT_EOS
        if not is_eos:
            ids.append(idx)
            if idx >= SANSKRIT_OFFSET:
                running_chars.append(DEVANAGARI[idx - SANSKRIT_OFFSET])

        frames.append({
            "phase": 1,
            "phase_step": len(frames),
            "chars": "".join(running_chars),
            "words": "",
            "entropy": entropy / np.log(vocab_size),
            "layers": self_layers,
            "self_rows": [last_query_from_attention_map(layer) for layer in self_layers],
            "hidden": hidden_vec,
            "trace": None,
        })
        if is_eos:
            break

    if not states:
        clear_capture()
        arr = torch.tensor([[SANSKRIT_BOS]], dtype=torch.long, device=device)
        hidden, _ = model.stage1(ids=arr, memory=prompt_memory, memory_pad=prompt_pad)
        states.append(hidden[:, -1:, :])
        frames.append({
            "phase": 1, "phase_step": 0, "chars": "", "words": "", "entropy": 0.0,
            "layers": [], "self_rows": [],
            "hidden": hidden[0, -1, :].cpu().numpy(), "trace": None,
        })

    stage1_memory = torch.cat(states, dim=1)
    return frames, stage1_memory


@torch.no_grad()
def run_stage2_frames(
    model, prompt_ids, stage1_memory, sanskrit_chars, tok, capture, clear_capture, args, device
):
    pad_id, bos_id, eos_id = tok.token_to_id(PAD), tok.token_to_id(BOS), tok.token_to_id(EOS)
    ids = [bos_id] + prompt_ids
    words = []
    frames = []
    num_layers = len(model.stage2.layers)
    vocab_size = model.english_vocab

    positional_capacity = int(model.stage2.pos.shape[1])
    max_generated_tokens = max(0, positional_capacity - len(ids) + 1)
    token_limit = min(int(args.infer_tokens), max_generated_tokens)
    if int(args.infer_tokens) > max_generated_tokens:
        print(
            f"warning: requested {args.infer_tokens} Stage 2 tokens, but the current "
            f"prompt/checkpoint combination supports at most {max_generated_tokens}; "
            f"limiting automatically",
            file=sys.stderr,
        )

    for step in range(token_limit):
        clear_capture()
        arr = torch.tensor([ids], dtype=torch.long, device=device)
        pad = torch.tensor([[x == pad_id for x in ids]], dtype=torch.bool, device=device)
        hidden, logits = model.stage2(ids=arr, memory=stage1_memory, self_pad=pad)
        last_hidden = hidden[0, -1, :].cpu().numpy()
        last_logits = logits[0, -1]
        forbidden = [pad_id, bos_id]
        if step < args.min_infer_tokens:
            forbidden.append(eos_id)
        probs = F.softmax(last_logits / args.stage2_temperature, dim=-1).clone()
        for f_id in forbidden:
            probs[f_id] = 0.0
        probs = probs / probs.sum().clamp_min(1e-9)
        entropy = float(-(probs * (probs + 1e-9).log()).sum())
        idx = sample_logits(
            last_logits, args.stage2_temperature, forbidden, args.stage2_top_k, args.stage2_top_p
        )

        self_layers = []
        for _, w in capture["stage2_self"][-min(3, num_layers):]:
            self_layers.extend(
                select_stable_head_maps(
                    attention_tensor_to_head_maps(w),
                    max_maps=max(1, int(getattr(args, "head_maps_per_tensor", 2))),
                )
            )
        cross_avg_layers = [
            collapse_attention_map(w) for _, w in capture["stage2_cross"][-num_layers:]
        ]
        cross_rows = (
            [last_query_from_attention_map(layer) for layer in cross_avg_layers]
            if cross_avg_layers
            else []
        )

        geom_layers = self_layers

        is_eos = idx == eos_id
        if not is_eos:
            ids.append(idx)
            words.append(tok.decode([idx]))

        frames.append({
            "phase": 2,
            "phase_step": len(frames),
            "chars": sanskrit_chars,
            "words": "".join(words).strip(),
            "entropy": entropy / np.log(vocab_size),
            "layers": geom_layers if geom_layers else (self_layers or cross_avg_layers),
            "self_rows": [last_query_from_attention_map(layer) for layer in self_layers],
            "hidden": last_hidden,
            "trace": cross_rows,
        })
        if is_eos:
            break

    return frames


def attention_tensor_to_head_maps(weights):
    if hasattr(weights, "detach"):
        arr = weights.detach().cpu().numpy()
    else:
        arr = np.asarray(weights)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 1:
        arr = np.sqrt(np.outer(arr, arr))[None, ...]
    else:
        arr = arr.reshape(1, 1, -1)

    head_maps = []
    for head in arr:
        head = np.asarray(head, dtype=np.float32)
        if head.ndim == 1:
            head = np.sqrt(np.outer(head, head))
        elif head.ndim != 2:
            head = head.reshape(head.shape[0], -1)
        head_maps.append(head.astype(np.float32))
    return head_maps


def collapse_attention_map(weights):
    maps = attention_tensor_to_head_maps(weights)
    if not maps:
        return np.zeros((1, 1), dtype=np.float32)
    if len(maps) == 1:
        return maps[0]
    stack = np.stack(maps, axis=0)
    return stack.mean(axis=0).astype(np.float32)


def select_stable_head_maps(head_maps, max_maps=2):
    if not head_maps or max_maps <= 0:
        return []
    count = min(int(max_maps), len(head_maps))
    if count == len(head_maps):
        indices = list(range(len(head_maps)))
    elif count == 1:
        indices = [0]
    else:
        indices = np.linspace(0, len(head_maps) - 1, count).round().astype(int).tolist()
    return [np.asarray(head_maps[i], dtype=np.float32) for i in indices]


def last_query_from_attention_map(attn_2d):
    arr = np.asarray(attn_2d, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[-1]
    return arr.reshape(-1)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def normalise_vector(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [3.0, 97.0])
    if float(hi - lo) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def pca_hidden_positions(hidden_matrix, extent, z_scale=0.32, seed=7):
    H = np.asarray(hidden_matrix, dtype=np.float32)
    n = H.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    centred = H - H.mean(axis=0, keepdims=True)
    if n == 1 or np.linalg.norm(centred) < 1e-8:
        coords = np.zeros((n, 3), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        dims = min(3, vt.shape[0])
        coords = centred @ vt[:dims].T
        if dims < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - dims)))

    trend = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    for axis in range(3):
        if float(np.dot(coords[:, axis], trend)) < 0.0:
            coords[:, axis] *= -1.0

    xy_radius = np.linalg.norm(coords[:, :2], axis=1)
    xy_scale = float(np.percentile(xy_radius, 96.0)) + 1e-6
    coords[:, :2] = np.clip(coords[:, :2] / xy_scale, -1.45, 1.45) * extent * 0.82

    z_den = float(np.percentile(np.abs(coords[:, 2]), 96.0)) + 1e-6
    coords[:, 2] = np.tanh(coords[:, 2] / z_den) * extent * float(z_scale)

    rng = np.random.RandomState(seed + 9017)
    jitter = rng.normal(scale=max(extent, 1e-6) * 1e-5, size=(n, 2)).astype(np.float32)
    delaunay_points = coords[:, :2].copy() + jitter
    return coords.astype(np.float32), delaunay_points.astype(np.float32)


def orient2d(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def circumcircle_contains(a, b, c, p, eps=1e-10):
    ax, ay = float(a[0] - p[0]), float(a[1] - p[1])
    bx, by = float(b[0] - p[0]), float(b[1] - p[1])
    cx, cy = float(c[0] - p[0]), float(c[1] - p[1])
    det = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    orientation = orient2d(a, b, c)
    return det > eps if orientation > 0.0 else det < -eps


def delaunay_triangulate(points):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    centre = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 1e-3)
    super_points = np.array([
        centre + [-24.0 * span, -12.0 * span],
        centre + [0.0, 24.0 * span],
        centre + [24.0 * span, -12.0 * span],
    ], dtype=np.float64)
    work = np.vstack([points, super_points])
    super_ids = (n, n + 1, n + 2)
    triangles = [super_ids]

    for point_idx in range(n):
        p = work[point_idx]
        bad_indices = []
        edge_counts = {}
        for tri_idx, tri in enumerate(triangles):
            a, b, c = work[list(tri)]
            if circumcircle_contains(a, b, c, p):
                bad_indices.append(tri_idx)
                for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edge = (u, v) if u < v else (v, u)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

        if not bad_indices:
            continue
        bad_set = set(bad_indices)
        triangles = [tri for idx, tri in enumerate(triangles) if idx not in bad_set]
        boundary = [edge for edge, count in edge_counts.items() if count == 1]
        for u, v in boundary:
            if abs(orient2d(work[u], work[v], p)) < 1e-12:
                continue
            tri = (u, v, point_idx)
            if orient2d(work[u], work[v], p) < 0.0:
                tri = (v, u, point_idx)
            triangles.append(tri)

    unique = {}
    for tri in triangles:
        if any(v >= n for v in tri):
            continue
        if abs(orient2d(work[tri[0]], work[tri[1]], work[tri[2]])) < 1e-12:
            continue
        unique[tuple(sorted(tri))] = tri
    if not unique:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(list(unique.values()), dtype=np.int32)


def filter_triangles(points, triangles, edge_quantile=0.985, max_aspect=18.0):
    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    if triangles.size == 0:
        return triangles.reshape(0, 3)

    metrics = []
    for tri in triangles:
        p = points[tri]
        lengths = np.array([
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[2] - p[1]),
            np.linalg.norm(p[0] - p[2]),
        ], dtype=np.float32)
        area2 = abs(orient2d(p[0], p[1], p[2]))
        aspect = float(lengths.max() ** 2 / (area2 + 1e-9))
        metrics.append((float(lengths.max()), aspect))

    edge_cutoff = float(np.quantile([m[0] for m in metrics], np.clip(edge_quantile, 0.5, 1.0)))
    kept = [
        tri
        for tri, (edge, aspect) in zip(triangles, metrics)
        if edge <= edge_cutoff and aspect <= max_aspect
    ]
    if len(kept) < max(1, len(triangles) // 3):
        kept = [tri for tri, (_, aspect) in zip(triangles, metrics) if aspect <= max_aspect * 1.5]
    return np.asarray(kept, dtype=np.int32).reshape(-1, 3)


def build_knn_edges(hidden_matrix, k=3):
    H = np.asarray(hidden_matrix, dtype=np.float32)
    n = H.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-7)
    similarity = Hn @ Hn.T
    np.fill_diagonal(similarity, -np.inf)
    edges = set()
    k = max(1, min(int(k), n - 1))
    for i in range(n):
        neighbours = np.argpartition(-similarity[i], k - 1)[:k]
        neighbours = neighbours[np.argsort(-similarity[i, neighbours])]
        for j in neighbours:
            u, v = (i, int(j)) if i < int(j) else (int(j), i)
            if u != v:
                edges.add((u, v))

    for i in range(n - 1):
        edges.add((i, i + 1))
    return np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)


def node_base_density(points, hidden_matrix, k=6):
    points = np.asarray(points, dtype=np.float32)
    H = np.asarray(hidden_matrix, dtype=np.float32)
    n = points.shape[0]
    if n <= 1:
        return np.ones((n,), dtype=np.float32)

    spatial_dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(spatial_dist, np.inf)
    kk = max(1, min(int(k), n - 1))
    near_dist = np.partition(spatial_dist, kk - 1, axis=1)[:, :kk].mean(axis=1)
    spatial_density = normalise_vector(1.0 / (near_dist + 1e-6))

    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-7)
    sim = Hn @ Hn.T
    np.fill_diagonal(sim, -1.0)
    semantic = np.partition(sim, n - kk, axis=1)[:, -kk:].mean(axis=1)
    semantic_density = normalise_vector(semantic)
    return np.clip(0.62 * spatial_density + 0.38 * semantic_density, 0.0, 1.0).astype(np.float32)


def stable_fracture_directions(count, seed=7):
    rng = np.random.RandomState(seed + 19073)
    dirs = rng.normal(size=(count, 3)).astype(np.float32)
    dirs[:, 2] *= 0.55
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-7
    return dirs


def aggregate_attention_rows(rows, target_length, take_tail=False):
    target_length = max(0, int(target_length))
    if not rows or target_length == 0:
        blank = np.zeros((target_length,), dtype=np.float32)
        return blank, blank.copy()
    packed = []
    for row in rows:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        arr = arr[-target_length:] if take_tail else arr[:target_length]
        padded = np.zeros((target_length,), dtype=np.float32)
        padded[:arr.size] = arr
        packed.append(padded)
    stack = np.stack(packed, axis=0)
    return normalise_vector(stack.mean(axis=0)), normalise_vector(stack.std(axis=0))


def build_node_signals(fr, hidden_matrix, stage1_count, base_density, previous_occupancy, args):
    n = hidden_matrix.shape[0]
    phase_step = int(fr.get("phase_step", 0))
    reveal = np.zeros((n,), dtype=np.float32)
    attention = np.zeros((n,), dtype=np.float32)
    disagreement = np.zeros((n,), dtype=np.float32)

    if int(fr["phase"]) == 1:
        visible = min(stage1_count, phase_step + 1)
        reveal[:visible] = 1.0
        mean, spread = aggregate_attention_rows(fr.get("self_rows", []), visible, take_tail=False)
        attention[:visible] = mean
        disagreement[:visible] = spread
        current_idx = max(0, visible - 1)
    else:
        stage2_visible = min(n - stage1_count, phase_step + 1)
        reveal[:stage1_count + stage2_visible] = 1.0
        cross_mean, cross_spread = aggregate_attention_rows(
            fr.get("trace", []), stage1_count, take_tail=False
        )
        attention[:stage1_count] = np.maximum(attention[:stage1_count], cross_mean)
        disagreement[:stage1_count] = np.maximum(disagreement[:stage1_count], cross_spread)
        self_mean, self_spread = aggregate_attention_rows(
            fr.get("self_rows", []), stage2_visible, take_tail=True
        )
        attention[stage1_count:stage1_count + stage2_visible] = self_mean
        disagreement[stage1_count:stage1_count + stage2_visible] = self_spread
        current_idx = min(n - 1, stage1_count + max(0, stage2_visible - 1))

    H = np.asarray(hidden_matrix, dtype=np.float32)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-7)
    current = np.asarray(fr["hidden"], dtype=np.float32)
    current /= np.linalg.norm(current) + 1e-7
    similarity = normalise_vector(Hn @ current)

    activation = reveal * np.clip(0.48 * similarity + 0.52 * attention, 0.0, 1.0)
    raw_occ = reveal * np.clip(
        0.18 + 0.43 * base_density + 0.31 * activation + 0.08 * disagreement,
        0.0,
        1.0,
    )

    if previous_occupancy is None:
        binary = raw_occ >= args.occupancy_open_threshold
    else:
        binary = np.where(
            previous_occupancy >= 0.5,
            raw_occ >= args.occupancy_keep_threshold,
            raw_occ >= args.occupancy_open_threshold,
        )
    occupancy = reveal * np.clip(0.30 * raw_occ + 0.70 * binary.astype(np.float32), 0.0, 1.0)
    temperature = reveal * np.clip(
        0.68 * activation + 0.24 * attention + 0.08 * base_density, 0.0, 1.0
    )
    return (
        activation.astype(np.float32),
        disagreement.astype(np.float32),
        temperature.astype(np.float32),
        occupancy.astype(np.float32),
        current_idx,
    )


def deform_hidden_positions(
    base_positions, activation, disagreement, occupancy, fracture_dirs, entropy, extent, args
):
    positions = np.asarray(base_positions, dtype=np.float32).copy()
    active_weight = occupancy + 1e-4
    centre = (positions * active_weight[:, None]).sum(axis=0) / active_weight.sum()
    radial = positions - centre
    radial_dir = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-7)

    positions += radial_dir * (activation[:, None] * args.activation_expand * extent)
    positions[:, 2] += (activation - 0.28) * args.activation_lift * extent
    fracture_amount = disagreement * (0.22 + 0.78 * float(entropy))
    positions += fracture_dirs * (fracture_amount[:, None] * args.fracture_gain * extent)
    return positions.astype(np.float32)


def pack_mesh_vertices(positions, disagreement, temperature, occupancy):
    verts = np.concatenate([
        np.asarray(positions, dtype=np.float32),
        np.asarray(disagreement, dtype=np.float32)[:, None],
        np.asarray(temperature, dtype=np.float32)[:, None],
        np.asarray(occupancy, dtype=np.float32)[:, None],
    ], axis=1)
    return np.ascontiguousarray(verts.astype(np.float32))


def trace_targets_from_frame(fr, stage1_count, top_k=8, weight_cutoff=0.15):
    if int(fr.get("phase", 1)) != 2 or not fr.get("trace") or stage1_count <= 0:
        return []
    mean, _ = aggregate_attention_rows(fr["trace"], stage1_count, take_tail=False)
    if mean.size == 0 or float(mean.max()) <= 1e-8:
        return []
    weights = mean / (float(mean.max()) + 1e-8)
    top = np.argsort(weights)[-min(int(top_k), weights.size):]
    return [(int(i), float(weights[i])) for i in top if float(weights[i]) >= float(weight_cutoff)]


def build_graph_line_data(positions, edges, activation, occupancy, lift=0.003, threshold=0.24):
    if edges.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for u, v in edges:
        if min(float(occupancy[u]), float(occupancy[v])) < threshold:
            continue
        strength = 0.18 + 0.58 * float((activation[u] + activation[v]) * 0.5)
        p0 = positions[u].copy()
        p1 = positions[v].copy()
        p0[2] += lift
        p1[2] += lift
        rows.extend([[p0[0], p0[1], p0[2], strength], [p1[0], p1[1], p1[2], strength]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def build_trace_line_data(positions, current_idx, targets, lift=0.055, curve_segments=6):
    if not targets or current_idx < 0 or current_idx >= len(positions):
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    anchor = positions[current_idx].astype(np.float32).copy()
    anchor[2] += lift * 0.45
    segments = max(2, int(curve_segments))
    for target_idx, strength in targets:
        if target_idx < 0 or target_idx >= len(positions) or target_idx == current_idx:
            continue
        target = positions[target_idx].astype(np.float32).copy()
        target[2] += lift * 0.22
        midpoint = (anchor + target) * 0.5
        chord = target - anchor
        side = np.cross(chord, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        side /= np.linalg.norm(side) + 1e-7
        control = midpoint + np.array([0.0, 0.0, lift * (0.75 + 0.8 * strength)], dtype=np.float32)
        control += side * lift * 0.22 * (0.5 + strength)
        points = []
        for s in range(segments + 1):
            t = s / float(segments)
            p = (1.0 - t) ** 2 * anchor + 2.0 * (1.0 - t) * t * control + t ** 2 * target
            points.append(p)
        for p0, p1 in zip(points[:-1], points[1:]):
            rows.extend([[p0[0], p0[1], p0[2], strength], [p1[0], p1[1], p1[2], strength]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def _step_hash(step, seed, salt, mod):
    h = (int(step) * 2654435761 + int(seed) * 40503 + int(salt) * 2246822519) & 0xFFFFFFFF
    return h % mod if mod > 0 else 0


class CutState:
    def __init__(self, seed, res, drift_per_frame=0.012):
        self.seed = seed
        self.res = res
        self.drift = drift_per_frame
        self.roll = (0, 0)
        self.azimuth = _step_hash(0, seed, 71, 3600) / 3600.0 * 2.0 * np.pi
        self.was_above = False

    def update(self, step, entropy, cut_threshold):
        above = entropy is not None and entropy > cut_threshold
        cut = bool(above and not self.was_above)
        self.was_above = bool(above)

        if cut:

            direction = 1.0 if _step_hash(step, self.seed, 71, 2) else -1.0
            self.azimuth += direction * 0.38
        else:
            self.azimuth += self.drift

        return self.roll, self.azimuth, cut


def perspective(fovy_rad, aspect, near, far):
    f = 1.0 / np.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def camera_eye(azimuth, elevation, distance):
    return np.array([
        distance * np.cos(elevation) * np.cos(azimuth),
        distance * np.cos(elevation) * np.sin(azimuth),
        distance * np.sin(elevation),
    ], dtype=np.float32)

MESH_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in float in_shade;
in float in_temp;
in float in_occ;
uniform mat4 u_mvp;
out float v_shade_vs;
out float v_temp_vs;
out float v_occ_vs;
out vec3 v_world_vs;
void main() {
    v_shade_vs = in_shade;
    v_temp_vs = in_temp;
    v_occ_vs = in_occ;
    v_world_vs = in_pos;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

MESH_GEOMETRY_SHADER = """
#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in float v_shade_vs[];
in float v_temp_vs[];
in float v_occ_vs[];
in vec3 v_world_vs[];
uniform float u_occ_threshold;
out float v_shade;
out float v_temp;
out float v_occ;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    float occ = (v_occ_vs[0] + v_occ_vs[1] + v_occ_vs[2]) / 3.0;
    if (occ < u_occ_threshold) return;
    vec3 a = v_world_vs[1] - v_world_vs[0];
    vec3 b = v_world_vs[2] - v_world_vs[0];
    vec3 raw_n = cross(a, b);
    if (length(raw_n) < 0.001) return;
    vec3 n = normalize(raw_n);
    float avg_shade = (v_shade_vs[0] + v_shade_vs[1] + v_shade_vs[2]) / 3.0;
    float avg_temp = (v_temp_vs[0] + v_temp_vs[1] + v_temp_vs[2]) / 3.0;
    for (int i = 0; i < 3; i++) {
        v_shade = avg_shade;
        v_temp = avg_temp;
        v_occ = occ;
        v_normal = n;
        v_pos = v_world_vs[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""

MESH_FRAGMENT_SHADER = """
#version 330
in float v_shade;
in float v_temp;
in float v_occ;
in vec3 v_normal;
in vec3 v_pos;
uniform vec3 u_light_dir;
uniform vec3 u_fill_light_dir;
uniform vec3 u_view_pos;
uniform float u_hue_shift;
uniform float u_entropy;
out vec4 f_colour;

vec3 hue_rotate(vec3 c, float a) {
    float u = cos(a), w = sin(a);
    mat3 m = mat3(
        0.299 + 0.701*u + 0.168*w, 0.587 - 0.587*u + 0.330*w, 0.114 - 0.114*u - 0.497*w,
        0.299 - 0.299*u - 0.328*w, 0.587 + 0.413*u + 0.035*w, 0.114 - 0.114*u + 0.292*w,
        0.299 - 0.300*u + 1.250*w, 0.587 - 0.588*u - 1.050*w, 0.114 + 0.886*u - 0.203*w
    );
    return clamp(m * c, 0.0, 1.0);
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 L2 = normalize(u_fill_light_dir);
    vec3 V = normalize(u_view_pos - v_pos);
    if (dot(N, V) < 0.0) N = -N;

    float key = max(dot(N, L), 0.0);
    float fill = max(dot(N, L2), 0.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.8);
    float spec = pow(max(dot(reflect(-L, N), V), 0.0), 30.0);

    vec3 cold0 = vec3(0.022, 0.031, 0.039);
    vec3 cold1 = vec3(0.080, 0.130, 0.162);
    vec3 steel = vec3(0.155, 0.245, 0.285);
    vec3 earth = vec3(0.315, 0.255, 0.170);
    vec3 ember = vec3(0.665, 0.345, 0.145);
    vec3 coral = vec3(0.725, 0.245, 0.285);

    float signal = clamp(0.58 * v_temp + 0.27 * v_shade + 0.15 * u_entropy, 0.0, 1.0);
    float band = floor(signal * 4.999) / 4.0;
    vec3 material = cold0;
    if (band >= 0.25) material = cold1;
    if (band >= 0.50) material = steel;
    if (band >= 0.75) material = earth;
    material = mix(material, ember, smoothstep(0.74, 0.98, v_temp) * (0.14 + 0.24 * u_entropy));
    material = mix(material, coral, smoothstep(0.68, 0.98, v_shade) * (0.10 + 0.30 * u_entropy));
    material = hue_rotate(material, u_hue_shift);

    vec3 linear = material * (0.15 + 0.76 * key + 0.24 * fill);
    linear += vec3(0.12, 0.145, 0.16) * rim * 0.58;
    linear += vec3(0.17, 0.18, 0.19) * spec * 0.32;
    linear *= mix(0.78, 1.0, v_occ);
    linear = linear / (linear + vec3(1.0));
    vec3 srgb = pow(clamp(linear, 0.0, 1.0), vec3(1.0 / 2.2));
    f_colour = vec4(srgb, 1.0);
}
"""

LINE_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in float in_strength;
uniform mat4 u_mvp;
out float v_strength;
void main() {
    v_strength = in_strength;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in float v_strength;
uniform float u_kind;
out vec4 f_colour;
void main() {
    vec3 graph_colour = vec3(0.31, 0.39, 0.42);
    vec3 trace_colour = vec3(0.76, 0.81, 0.80);
    vec3 c = mix(graph_colour, trace_colour, u_kind);
    float alpha = mix(0.16 + 0.30 * v_strength, 0.48 + 0.46 * v_strength, u_kind);
    f_colour = vec4(c * (0.45 + 0.55 * v_strength), alpha);
}
"""


class MeshRenderer:
    def __init__(self, width, height, vertex_count, triangle_indices, gl_backend=None):
        self.width, self.height = width, height
        kwargs = {"backend": gl_backend} if gl_backend else {}
        self.ctx = moderngl.create_standalone_context(**kwargs)
        self.fbo = self.ctx.simple_framebuffer((width, height))

        self.mesh_prog = self.ctx.program(
            vertex_shader=MESH_VERTEX_SHADER,
            geometry_shader=MESH_GEOMETRY_SHADER,
            fragment_shader=MESH_FRAGMENT_SHADER,
        )
        self.line_prog = self.ctx.program(
            vertex_shader=LINE_VERTEX_SHADER,
            fragment_shader=LINE_FRAGMENT_SHADER,
        )

        indices = np.asarray(triangle_indices, dtype=np.int32).reshape(-1)
        if indices.size == 0:
            raise RuntimeError("Hidden-state topology produced no valid triangles")
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vbo = self.ctx.buffer(reserve=int(vertex_count) * 6 * 4)
        self.vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(self.vbo, "3f 1f 1f 1f", "in_pos", "in_shade", "in_temp", "in_occ")],
            self.ibo,
        )

    def _render_lines(self, line_data, mvp, kind, line_width):
        line_data = np.asarray(line_data, dtype=np.float32)
        if line_data.size == 0:
            return
        vbo = self.ctx.buffer(np.ascontiguousarray(line_data).tobytes())
        vao = self.ctx.vertex_array(
            self.line_prog,
            [(vbo, "3f 1f", "in_pos", "in_strength")],
        )
        self.line_prog["u_mvp"].write(np.ascontiguousarray(mvp.T).tobytes())
        self.line_prog["u_kind"].value = float(kind)
        try:
            self.ctx.line_width = float(line_width)
        except Exception:
            pass
        vao.render(moderngl.LINES)
        vao.release()
        vbo.release()

    def render_frame(self, vertex_data, mvp, light_dir, fill_light_dir, eye, hue_shift,
                     entropy, occ_threshold, graph_lines, trace_lines,
                     graph_line_width=1.0, trace_line_width=1.8):
        self.fbo.use()
        self.fbo.clear(0.008, 0.011, 0.012, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)

        self.vbo.write(np.ascontiguousarray(vertex_data))
        self.mesh_prog["u_mvp"].write(np.ascontiguousarray(mvp.T).tobytes())
        self.mesh_prog["u_light_dir"].value = tuple(float(x) for x in light_dir)
        self.mesh_prog["u_fill_light_dir"].value = tuple(float(x) for x in fill_light_dir)
        self.mesh_prog["u_view_pos"].value = tuple(float(x) for x in eye)
        self.mesh_prog["u_hue_shift"].value = float(hue_shift)
        self.mesh_prog["u_entropy"].value = float(entropy)
        self.mesh_prog["u_occ_threshold"].value = float(occ_threshold)
        self.vao.render(moderngl.TRIANGLES)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._render_lines(graph_lines, mvp, kind=0.0, line_width=graph_line_width)
        self._render_lines(trace_lines, mvp, kind=1.0, line_width=trace_line_width)
        self.ctx.disable(moderngl.BLEND)

        data = self.fbo.read(components=3, alignment=1)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.flipud(frame)


def _split_overlong_token(token, font, max_width, draw):
    if not token:
        return []
    pieces = []
    current = ""
    for char in token:
        trial = current + char
        if current and draw.textlength(trial, font=font) > max_width:
            pieces.append(current)
            current = char
        else:
            current = trial
    if current:
        pieces.append(current)
    return pieces


def wrap_text(text, font, max_width, draw):
    if not text:
        return []

    lines = []
    paragraphs = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        tokens = paragraph.split()

        current = ""
        for token in tokens:
            token_parts = (
                _split_overlong_token(token, font, max_width, draw)
                if draw.textlength(token, font=font) > max_width
                else [token]
            )
            for part_index, part in enumerate(token_parts):
                separator = " " if current and part_index == 0 else ""
                trial = current + separator + part
                if current and draw.textlength(trial, font=font) > max_width:
                    lines.append(current)
                    current = part
                else:
                    current = trial

                if part_index < len(token_parts) - 1:
                    lines.append(current)
                    current = ""

        if current:
            lines.append(current)
        if paragraph_index < len(paragraphs) - 1 and lines and lines[-1] != "":
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def visible_line_window(lines, line_height, max_height, prefer_tail=True):
    if not lines or max_height <= 0:
        return [], False
    max_lines = max(1, int(max_height // max(1, line_height)))
    if len(lines) <= max_lines:
        return list(lines), False
    selected = list(lines[-max_lines:] if prefer_tail else lines[:max_lines])
    if selected:
        if prefer_tail:
            selected[0] = "… " + selected[0].lstrip("… ")
        else:
            selected[-1] = selected[-1].rstrip("… ") + " …"
    return selected, True


def glitch_slice_shift(img, x0, y0, x1, y1, step, seed, intensity):
    region = img[y0:y1, x0:x1].copy()
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return
    n_bands = 3 + int(6 * intensity)
    band_h = max(1, h // n_bands)
    for b in range(n_bands):
        by0 = b * band_h
        by1 = min(h, by0 + band_h)
        if by1 <= by0:
            continue
        shift = _step_hash(step, seed, 3000 + b, w) - w // 2
        shift = int(shift * intensity)
        if shift == 0:
            continue
        region[by0:by1] = np.roll(region[by0:by1], shift, axis=1)
    img[y0:y1, x0:x1] = region


def composite_text(
    frame_rgb,
    chars,
    words,
    entropy,
    phase,
    deva_font,
    latin_font,
    fallback_font,
    width,
    step,
    seed,
    glitch_threshold=0.6,
    text_width_scale=0.40,
    text_height_scale=0.72,
    deva_height_scale=0.24,
    text_font_scale=1.0,
):
    img = Image.fromarray(frame_rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    pad = max(16, width // 44)
    paragraph_w = min(int(width * text_width_scale), 580)

    font_scale = max(0.65, float(text_font_scale))
    deva_size = max(20, int((width // 48) * font_scale))
    eng_size = max(14, int((width // 66) * font_scale))
    cap_size = max(10, int((width // 108) * font_scale))

    latin_font_path = latin_font or fallback_font
    deva_font_path = deva_font or latin_font_path or fallback_font

    f_deva = ImageFont.truetype(deva_font_path, deva_size)
    f_eng = ImageFont.truetype(latin_font_path, eng_size)
    f_cap = ImageFont.truetype(latin_font_path, cap_size)

    deva_text = str(chars or "").strip()
    eng_text = " ".join(str(words or "").split())
    deva_lines = wrap_text(deva_text, f_deva, paragraph_w, draw)
    eng_lines = wrap_text(eng_text, f_eng, paragraph_w, draw)

    line_h_deva = max(1, int(deva_size * 1.18))
    line_h_eng = max(1, int(eng_size * 1.28))
    section_gap = max(6, eng_size // 3)

    caption_y = img.height - cap_size - 11
    text_region_bottom = min(caption_y - 12, int(img.height * text_height_scale))
    available_height = max(1, text_region_bottom - pad)

    if deva_lines and eng_lines:
        deva_budget = min(
            int(img.height * deva_height_scale),
            max(line_h_deva, int(available_height * 0.40)),
        )
        eng_budget = max(line_h_eng, available_height - deva_budget - section_gap)
    elif deva_lines:
        deva_budget = available_height
        eng_budget = 0
    else:
        deva_budget = 0
        eng_budget = available_height

    visible_deva, _ = visible_line_window(
        deva_lines, line_h_deva, deva_budget, prefer_tail=True
    )
    visible_eng, _ = visible_line_window(
        eng_lines, line_h_eng, eng_budget, prefer_tail=True
    )

    deva_colour = (140, 150, 148, 205)
    eng_colour = (182, 186, 182, 190)
    cap_colour = (104, 108, 106, 180)

    def draw_shadowed_text(x, y, text, font, fill, shadow=(0, 0, 0, 120), offset=1):
        draw.text((x + offset, y + offset), text, font=font, fill=shadow)
        draw.text((x, y), text, font=font, fill=fill)

    text_top = pad
    y = pad
    for line in visible_deva:
        if line:
            draw_shadowed_text(pad, y, line, f_deva, deva_colour)
        y += line_h_deva

    if visible_deva and visible_eng:
        y += section_gap

    for line in visible_eng:
        if line:
            draw_shadowed_text(pad, y, line, f_eng, eng_colour)
        y += line_h_eng

    text_bottom = min(y, text_region_bottom)

    caption = f"p{phase}  H={entropy:.2f}"
    draw_shadowed_text(
        pad,
        caption_y,
        caption,
        f_cap,
        cap_colour,
        shadow=(0, 0, 0, 90),
        offset=1,
    )

    out = np.array(img.convert("RGB"))
    if entropy > glitch_threshold and text_bottom > text_top:
        intensity = min(
            1.0,
            (entropy - glitch_threshold) / max(1e-6, 1.0 - glitch_threshold),
        )
        x0, y0 = 0, max(0, text_top - 4)
        x1 = min(out.shape[1], pad + paragraph_w + pad)
        y1 = min(out.shape[0], text_bottom + 4)
        glitch_slice_shift(out, x0, y0, x1, y1, step, seed, intensity)
    return out


def build_hue_projector(d_model, seed=1234):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(d_model,)).astype(np.float32)


def hue_shift_from_hidden(hidden_vec, proj, max_rad=0.21):
    denom = (np.linalg.norm(hidden_vec) * np.linalg.norm(proj)) + 1e-8
    cos = float(np.dot(hidden_vec, proj) / denom)
    return cos * max_rad


def render(stage1_frames, stage2_frames, args):
    deva_font = find_font(_DEVA_FONTS)
    latin_font = find_font(LATIN_TEXT_FONTS) or find_fallback_font()
    fallback_font = find_fallback_font()

    model_frames = stage1_frames + stage2_frames
    if len(model_frames) < 3:
        raise RuntimeError(
            "At least three generated states are required to build the hidden-state mesh"
        )

    hidden_matrix = np.stack([fr["hidden"] for fr in model_frames], axis=0).astype(np.float32)
    stage1_count = len(stage1_frames)
    base_positions, delaunay_points = pca_hidden_positions(
        hidden_matrix, args.extent, z_scale=args.base_z_scale, seed=args.seed
    )
    triangles = delaunay_triangulate(delaunay_points)
    triangles = filter_triangles(
        delaunay_points, triangles,
        edge_quantile=args.triangle_edge_quantile,
        max_aspect=args.triangle_max_aspect,
    )
    if triangles.size == 0:
        raise RuntimeError(
            "Delaunay triangulation failed; try a different seed or a longer generation"
        )

    knn_edges = build_knn_edges(hidden_matrix, k=args.knn_k)
    density = node_base_density(delaunay_points, hidden_matrix, k=max(4, args.knn_k + 2))
    fracture_dirs = stable_fracture_directions(len(model_frames), seed=args.seed)

    print(
        f"topology: {len(model_frames)} hidden-state vertices, "
        f"{len(triangles)} Delaunay facets, {len(knn_edges)} k-NN/spine struts",
        file=sys.stderr,
    )

    renderer = MeshRenderer(
        args.width, args.height, len(model_frames), triangles, args.gl_backend
    )
    d_model = hidden_matrix.shape[1]
    hue_projector = build_hue_projector(d_model, seed=args.seed)
    cut_state = CutState(
        seed=args.seed,
        res=max(1, len(model_frames)),
        drift_per_frame=args.azimuth_drift,
    )
    aspect = args.width / args.height

    prepared = []
    prev_positions = None
    prev_activation = None
    prev_disagreement = None
    prev_temperature = None
    prev_occupancy = None

    for n, fr in enumerate(model_frames):
        activation, disagreement, temperature, occupancy, current_idx = build_node_signals(
            fr, hidden_matrix, stage1_count, density, prev_occupancy, args
        )
        positions = deform_hidden_positions(
            base_positions, activation, disagreement, occupancy, fracture_dirs,
            fr["entropy"], args.extent, args,
        )

        _, azimuth, cut = cut_state.update(n, fr["entropy"], args.cut_entropy_threshold)
        if prev_positions is not None:
            settle = 1.0 - min(1.0, float(fr["entropy"]))
            inertia = args.temporal_inertia * (0.56 + 0.44 * settle)
            if cut:
                inertia *= args.cut_inertia_scale
            positions = lerp(positions, prev_positions, inertia)
            activation = lerp(activation, prev_activation, inertia)
            disagreement = lerp(disagreement, prev_disagreement, inertia)
            temperature = lerp(temperature, prev_temperature, inertia)
            occupancy = lerp(occupancy, prev_occupancy, inertia * 0.55)

        prev_positions = positions.copy()
        prev_activation = activation.copy()
        prev_disagreement = disagreement.copy()
        prev_temperature = temperature.copy()
        prev_occupancy = occupancy.copy()

        prepared.append({
            "fr": fr,
            "positions": positions,
            "activation": activation,
            "disagreement": disagreement,
            "temperature": temperature,
            "occupancy": occupancy,
            "current_idx": current_idx,
            "trace_targets": trace_targets_from_frame(
                fr, stage1_count, top_k=args.trace_top_k,
                weight_cutoff=args.trace_weight_cutoff,
            ),
            "azimuth": azimuth,
            "hue": hue_shift_from_hidden(fr["hidden"], hue_projector),
            "cut": cut,
        })

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", args.ffmpeg_preset,
        "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        str(args.video_out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    key_light = np.array([0.38, -0.54, 0.75], dtype=np.float32)
    fill_light = np.array([-0.55, 0.45, 0.52], dtype=np.float32)
    total = 0
    previous = None

    def emit(state, t=None):
        nonlocal total
        if previous is None or t is None or state["cut"]:
            interp = state
        else:
            interp = dict(state)
            for key in ("positions", "activation", "disagreement", "temperature", "occupancy"):
                interp[key] = lerp(previous[key], state[key], t)
            interp["azimuth"] = float(lerp(previous["azimuth"], state["azimuth"], t))
            interp["hue"] = float(lerp(previous["hue"], state["hue"], t))

        eye = camera_eye(interp["azimuth"], args.elevation, args.camera_distance)
        view = look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
        proj_mat = perspective(np.radians(args.fov), aspect, 0.1, 10.0)
        mvp = proj_mat @ view

        vertex_data = pack_mesh_vertices(
            interp["positions"], interp["disagreement"], interp["temperature"], interp["occupancy"]
        )
        graph_lines = build_graph_line_data(
            interp["positions"], knn_edges, interp["activation"], interp["occupancy"],
            lift=args.graph_lift, threshold=args.graph_occupancy_threshold,
        )
        trace_lines = build_trace_line_data(
            interp["positions"], interp["current_idx"], interp["trace_targets"],
            lift=args.trace_lift, curve_segments=args.trace_curve_segments,
        )

        gl_frame = renderer.render_frame(
            vertex_data, mvp, key_light, fill_light, eye, interp["hue"],
            interp["fr"]["entropy"], args.occupancy_render_threshold,
            graph_lines, trace_lines,
            graph_line_width=args.graph_line_width,
            trace_line_width=args.trace_line_width,
        )
        final = composite_text(
            gl_frame, interp["fr"]["chars"], interp["fr"]["words"],
            interp["fr"]["entropy"], interp["fr"]["phase"],
            deva_font, latin_font, fallback_font, args.width,
            step=total,
            seed=args.seed,
            glitch_threshold=args.text_glitch_threshold,
            text_width_scale=args.text_width_scale,
            text_height_scale=args.text_height_scale,
            deva_height_scale=args.text_deva_height_scale,
            text_font_scale=args.text_font_scale,
        )
        proc.stdin.write(np.ascontiguousarray(final).tobytes())
        total += 1

    for idx, state in enumerate(prepared):
        if previous is not None and not state["cut"] and args.subframes > 0:
            for s in range(1, args.subframes + 1):
                emit(state, t=s / float(args.subframes + 1))
        emit(state)
        previous = state
        if (idx + 1) % 20 == 0 or idx + 1 == len(prepared):
            tag = "CUT" if state["cut"] else "   "
            print(
                f"rendered {idx + 1}/{len(prepared)} model frames -> "
                f"{total} video frames [{tag}]",
                file=sys.stderr,
            )

    proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"wrote {total} frames to {args.video_out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--prompt", required=True)
    p.add_argument("--video-out", default="kavi_interior.mp4")
    p.add_argument("--device", default="auto")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    p.add_argument(
        "--extent",
        type=float,
        default=0.78,
        help="half-width used to scale the hidden-state object",
    )
    p.add_argument(
        "--base-z-scale",
        type=float,
        default=0.34,
        help="depth assigned to the third hidden-state principal component",
    )
    p.add_argument(
        "--knn-k", type=int, default=3, help="semantic neighbours per hidden-state vertex"
    )
    p.add_argument(
        "--triangle-edge-quantile",
        type=float,
        default=0.985,
        help="remove abnormally long Delaunay facets",
    )
    p.add_argument(
        "--triangle-max-aspect",
        type=float,
        default=18.0,
        help="maximum thinness of retained facets",
    )
    p.add_argument(
        "--activation-expand",
        type=float,
        default=0.095,
        help="radial expansion driven by current-state similarity/attention",
    )
    p.add_argument(
        "--activation-lift",
        type=float,
        default=0.30,
        help="vertical activation displacement",
    )
    p.add_argument(
        "--fracture-gain",
        type=float,
        default=0.17,
        help="entropy-scaled separation driven by head/layer disagreement",
    )

    p.add_argument("--fov", type=float, default=33.0, help="camera vertical FOV, degrees")
    p.add_argument("--camera-distance", type=float, default=2.55)
    p.add_argument("--elevation", type=float, default=0.31, help="camera elevation, radians")
    p.add_argument(
        "--azimuth-drift",
        type=float,
        default=0.0035,
        help="slow camera drift per model state",
    )
    p.add_argument("--cut-entropy-threshold", type=float, default=0.57)
    p.add_argument("--temporal-inertia", type=float, default=0.76)
    p.add_argument("--cut-inertia-scale", type=float, default=0.30)

    p.add_argument("--occupancy-open-threshold", type=float, default=0.43)
    p.add_argument("--occupancy-keep-threshold", type=float, default=0.30)
    p.add_argument("--occupancy-render-threshold", type=float, default=0.34)
    p.add_argument("--graph-occupancy-threshold", type=float, default=0.24)
    p.add_argument("--graph-lift", type=float, default=0.003)
    p.add_argument("--trace-lift", type=float, default=0.060)
    p.add_argument("--graph-line-width", type=float, default=1.0)
    p.add_argument("--trace-line-width", type=float, default=1.8)
    p.add_argument("--trace-curve-segments", type=int, default=6)
    p.add_argument("--trace-top-k", type=int, default=8)
    p.add_argument("--trace-weight-cutoff", type=float, default=0.15)

    p.add_argument("--text-glitch-threshold", type=float, default=0.64)
    p.add_argument(
        "--text-width-scale", type=float, default=0.40,
        help="fraction of frame width available to the wrapped text paragraph",
    )
    p.add_argument(
        "--text-height-scale", type=float, default=0.72,
        help="vertical frame position below which paragraph text will not draw",
    )
    p.add_argument(
        "--text-deva-height-scale", type=float, default=0.24,
        help="maximum frame-height share reserved for the Sanskrit paragraph",
    )
    p.add_argument(
        "--text-font-scale", type=float, default=1.0,
        help="global scale for Sanskrit, English, and diagnostic typography",
    )
    p.add_argument("--head-maps-per-tensor", type=int, default=2)
    p.add_argument("--subframes", type=int, default=2)
    p.add_argument("--ffmpeg-preset", default="slow")
    p.add_argument("--crf", type=int, default=15)
    p.add_argument("--gl-backend", default=None)

    p.add_argument("--stage1-infer-chars", type=int, default=None)
    p.add_argument("--stage1-temperature", type=float, default=0.92)
    p.add_argument("--stage2-temperature", type=float, default=0.82)
    p.add_argument("--stage1-top-k", type=int, default=24)
    p.add_argument("--stage2-top-k", type=int, default=32)
    p.add_argument("--stage1-top-p", type=float, default=0.94)
    p.add_argument("--stage2-top-p", type=float, default=0.95)
    p.add_argument("--infer-tokens", type=int, default=80)
    p.add_argument("--min-infer-tokens", type=int, default=0)
    p.add_argument("--soft-stage1-geometry", dest="soft_stage1_geometry", action="store_true")
    p.add_argument("--hard-stage1-geometry", dest="soft_stage1_geometry", action="store_false")
    p.set_defaults(soft_stage1_geometry=True)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--mesh-res", type=int, default=72, help=argparse.SUPPRESS)
    p.add_argument("--void-radius-scale", type=float, default=1.18, help=argparse.SUPPRESS)
    p.add_argument("--height-curve", type=float, default=0.78, help=argparse.SUPPRESS)
    p.add_argument("--displacement-gain", type=float, default=1.12, help=argparse.SUPPRESS)
    p.add_argument("--ridge-gain", type=float, default=0.18, help=argparse.SUPPRESS)
    p.add_argument("--lateral-gain", type=float, default=0.055, help=argparse.SUPPRESS)
    p.add_argument("--latent-modes", type=int, default=14, help=argparse.SUPPRESS)
    p.add_argument("--attention-geometry-mix", type=float, default=0.24, help=argparse.SUPPRESS)
    p.add_argument("--activation-geometry-mix", type=float, default=0.46, help=argparse.SUPPRESS)
    p.add_argument("--body-sigma", type=float, default=0.17, help=argparse.SUPPRESS)
    p.add_argument("--spatial-smooth-passes", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--base-relief", type=float, default=0.14, help=argparse.SUPPRESS)
    p.add_argument("--entropy-relief-gain", type=float, default=0.24, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    p = paths(args)
    device = pick_device(args.device)
    config = json.loads(p["config"].read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(resolve_artifact_path(config["tokeniser"], p["tokeniser"])))
    model_args = config_args(config)
    model = TwoStageGenerator(config["english_vocab"], model_args).to(device)
    weights_path = resolve_artifact_path(config["weights"], p["weights"])
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    capture, clear_capture, _ = attach_capture(model)

    prompt_ids = tok.encode(args.prompt).ids[-model_args.max_prompt_tokens:]
    if not prompt_ids:
        prompt_ids = [tok.token_to_id(BOS)]
    prompt_memory, prompt_pad = model.encode_prompt(prompt_ids, tok.token_to_id(PAD), device)

    stage1_frames, stage1_memory = run_stage1_frames(
        model, prompt_memory, prompt_pad, capture, clear_capture, args, device
    )
    sanskrit_chars = stage1_frames[-1]["chars"]

    stage2_frames = run_stage2_frames(
        model, prompt_ids, stage1_memory, sanskrit_chars, tok, capture, clear_capture, args, device
    )

    if args.soft_stage1_geometry and hasattr(model, "soft_stage1_infer") and stage1_frames:
        soft_memory = model.soft_stage1_infer(prompt_memory, prompt_pad, len(stage1_frames))
        soft_states = soft_memory[0].detach().cpu().numpy()
        for idx in range(min(len(stage1_frames), soft_states.shape[0])):
            stage1_frames[idx]["hidden"] = soft_states[idx].astype(np.float32)
        print(
            "geometry: using soft Stage 1 trajectory; sampled script remains unchanged",
            file=sys.stderr,
        )

    print(sanskrit_chars)
    print(stage2_frames[-1]["words"] if stage2_frames else "")

    render(stage1_frames, stage2_frames, args)


if __name__ == "__main__":
    main()
