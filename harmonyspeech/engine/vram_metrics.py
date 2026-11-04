"""Prometheus text rendering for VRAM metrics (Phase 4-5).

The engine owns the accounting (:meth:`HarmonySpeechEngine.get_vram_stats`); this
module is a pure renderer so it can be unit-tested without a server or a GPU.
"""

from __future__ import annotations

_HELP = {
    "hse_model_vram_bytes": "CUDA allocation attributable to one resident model, in bytes.",
    "hse_model_weights_bytes": "Resident parameter/buffer size of one model (host or device), in bytes.",
    "hse_vram_allocated_bytes": "torch.cuda.memory_allocated() at scrape time.",
    "hse_vram_reserved_bytes": "torch.cuda.memory_reserved() at scrape time.",
    "hse_vram_peak_allocated_bytes": "Peak torch.cuda.memory_allocated() since boot.",
    "hse_vram_total_bytes": "Total memory of the active CUDA device.",
    "hse_vram_headroom_bytes": "Free device memory (total minus reserved).",
    "hse_resident_models": "Number of model executors currently resident.",
    "hse_shared_backbone_bytes": "Bytes of deduplicated backbone weights per backbone_group.",
    "hse_shared_total_bytes": "Bytes held by the shared-weight registry (loaded once).",
    "hse_shared_components": "Number of distinct shared components in the registry.",
}


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sample(name: str, value, labels: dict[str, str] | None = None) -> str:
    if labels:
        label_text = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
        return f"{name}{{{label_text}}} {int(value)}"
    return f"{name} {int(value)}"


def render_vram_metrics(stats: dict) -> str:
    """Render :func:`get_vram_stats` output as Prometheus text exposition."""
    lines: list[str] = []
    emitted: set[str] = set()

    def header(name: str) -> None:
        if name in emitted:
            return
        emitted.add(name)
        lines.append(f"# HELP {name} {_HELP.get(name, name)}")
        lines.append(f"# TYPE {name} gauge")

    models = stats.get("models") or {}
    if models:
        header("hse_model_vram_bytes")
        for name, detail in models.items():
            lines.append(
                _sample(
                    "hse_model_vram_bytes",
                    detail.get("vram_bytes", 0),
                    {"model": name, "model_type": detail.get("model_type") or "unknown"},
                )
            )
        header("hse_model_weights_bytes")
        for name, detail in models.items():
            lines.append(_sample("hse_model_weights_bytes", detail.get("weights_bytes", 0), {"model": name}))

    for metric_name, key in (
        ("hse_vram_allocated_bytes", "allocated_bytes"),
        ("hse_vram_reserved_bytes", "reserved_bytes"),
        ("hse_vram_peak_allocated_bytes", "peak_allocated_bytes"),
        ("hse_vram_total_bytes", "total_bytes"),
        ("hse_vram_headroom_bytes", "headroom_bytes"),
        ("hse_resident_models", "resident_models"),
    ):
        header(metric_name)
        lines.append(_sample(metric_name, stats.get(key, 0) or 0))

    groups = stats.get("groups") or {}
    header("hse_shared_backbone_bytes")
    if groups:
        for group, value in groups.items():
            lines.append(_sample("hse_shared_backbone_bytes", value, {"group": group}))
    else:
        # Prometheus requires at least one sample for a declared metric family;
        # render an explicit zero so dashboards alarm on "no sharing" honestly.
        lines.append(_sample("hse_shared_backbone_bytes", 0, {"group": "none"}))

    shared = stats.get("shared") or {}
    header("hse_shared_total_bytes")
    lines.append(_sample("hse_shared_total_bytes", shared.get("total_bytes", 0) or 0))
    header("hse_shared_components")
    lines.append(_sample("hse_shared_components", shared.get("entries", 0) or 0))

    return "\n".join(lines) + "\n"


__all__ = ["render_vram_metrics"]
