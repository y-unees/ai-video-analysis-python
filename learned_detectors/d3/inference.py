from __future__ import annotations

import math
from typing import Any


def first_order_values(embeddings: list[list[float]], distance_mode: str) -> list[float]:
    values: list[float] = []
    for first, second in zip(embeddings, embeddings[1:]):
        if distance_mode == "l2":
            values.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second))))
        elif distance_mode == "cos":
            dot = sum(a * b for a, b in zip(first, second))
            norm_a = math.sqrt(sum(a * a for a in first))
            norm_b = math.sqrt(sum(b * b for b in second))
            values.append(dot / max(norm_a * norm_b, 1e-12))
        else:
            raise ValueError(f"Unsupported distance mode: {distance_mode}")
    return values


def second_order_values(first_order: list[float]) -> list[float]:
    return [current - previous for previous, current in zip(first_order, first_order[1:])]


def summarize_second_order(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "second_order_value_count": 0,
            "second_order_mean": None,
            "second_order_standard_deviation": None,
            "minimum_second_order_value": None,
            "maximum_second_order_value": None,
        }
    mean = sum(values) / len(values)
    denominator = len(values) - 1 if len(values) > 1 else 1
    variance = sum((value - mean) ** 2 for value in values) / denominator
    return {
        "second_order_value_count": len(values),
        "second_order_mean": mean,
        "second_order_standard_deviation": math.sqrt(variance),
        "minimum_second_order_value": min(values),
        "maximum_second_order_value": max(values),
    }


def upstream_second_order_summary_from_features(
    embeddings: list[list[float]],
    distance_mode: str,
) -> tuple[list[float], list[float], dict[str, float | int | None]]:
    first_order = first_order_values(embeddings, distance_mode)
    second_order = second_order_values(first_order)
    return first_order, second_order, summarize_second_order(second_order)


def validate_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def torch_forward_score(model: Any, frames_tensor: Any) -> tuple[Any, list[float], list[float], dict[str, Any]]:
    import torch

    with torch.no_grad():
        embeddings, second_mean, second_std = model(frames_tensor)
    embeddings_list = embeddings.detach().cpu().reshape(embeddings.shape[1], -1).tolist()
    first_order = first_order_values(embeddings_list, model.loss_type)
    second_order = second_order_values(first_order)
    summary = summarize_second_order(second_order)
    summary["official_torch_second_order_mean"] = float(second_mean.detach().cpu().flatten()[0])
    summary["official_torch_second_order_standard_deviation"] = float(second_std.detach().cpu().flatten()[0])
    return embeddings, first_order, second_order, summary
