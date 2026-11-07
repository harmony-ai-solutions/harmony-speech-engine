"""Measure the engine's VRAM footprint (Phase 4-5).

Boots the engine from a config, waits for readiness, samples steady-state VRAM
(torch allocation/reservation + ``nvidia-smi``), optionally runs one warmup
request per configured toolchain to capture an activation peak, prints a table
and optionally appends it to ``docs/benchmarks/vram-profiles.md``.

Usage
-----
    python scripts/measure_vram.py --config config.gpu.yml \
        --label "fp16 + dedup (Phase 4)" \
        --json-out out/vram/after.json \
        --append docs/benchmarks/vram-profiles.md

    # skip request warmup (boot-only weights measurement, much faster)
    python scripts/measure_vram.py --config config.gpu.yml --no-warmup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

GIB = float(2**30)

#: Toolchains we try to warm up, in order. Only families actually configured and
#: advertised by the engine are exercised.
WARMUP_FAMILIES = (
    "chatterbox",
    "chatterbox_turbo",
    "chatterbox_multilingual",
    "melotts",
    "openvoice_v1",
    "openvoice_v2",
    "harmonyspeech",
    "kittentts",
    "voicefixer",
)
WARMUP_TEXT = "Short warmup sentence for the VRAM profile."


def gib(value: int | float | None) -> float:
    return round(float(value or 0) / GIB, 3)


def query_nvidia_smi(device_index: int = 0) -> dict:
    """Return ``{"used_mib", "total_mib"}`` from nvidia-smi (0 when unavailable)."""
    result = {"used_mib": 0, "total_mib": 0}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            first = proc.stdout.strip().splitlines()[0]
            used, total = [part.strip() for part in first.split(",")]
            result["used_mib"] = int(float(used))
            result["total_mib"] = int(float(total))
    except Exception:  # pragma: no cover - depends on host tooling
        pass
    return result


def collect_vram_stats(engine) -> dict:
    """Read the engine's VRAM stats, tolerating shell/mock engines."""
    stats = engine.get_vram_stats()
    if not isinstance(stats, dict):
        raise TypeError(f"get_vram_stats() returned {type(stats).__name__}")
    return stats


def format_table(stats: dict, smi: dict | None = None) -> str:
    """Render a human table from :func:`collect_vram_stats` output."""
    lines = [
        f"{'model':<28} {'device':<7} {'dtype':<10} {'group':<20} {'vram MiB':>10} {'weights MiB':>12}",
        "-" * 92,
    ]
    for name, detail in stats.get("models", {}).items():
        lines.append(
            f"{name:<28} {str(detail.get('device') or '-'):<7} "
            f"{str(detail.get('dtype') or '-').replace('torch.', ''):<10} "
            f"{str(detail.get('backbone_group') or '-'):<20} "
            f"{gib(detail.get('vram_bytes')) * 1024:>10.0f} "
            f"{gib(detail.get('weights_bytes')) * 1024:>12.0f}"
        )
    lines.append("-" * 92)
    lines.append(
        f"resident models: {stats.get('resident_models', 0)} · "
        f"torch allocated {gib(stats.get('allocated_bytes')):.2f} GiB · "
        f"reserved {gib(stats.get('reserved_bytes')):.2f} GiB · "
        f"peak {gib(stats.get('peak_allocated_bytes')):.2f} GiB"
    )
    total = stats.get("total_bytes") or 0
    if total:
        lines.append(
            f"device total {gib(total):.2f} GiB · "
            f"device used {gib(stats.get('device_used_bytes')):.2f} GiB · "
            f"headroom (device free) {gib(stats.get('headroom_bytes')):.2f} GiB"
        )
    groups = stats.get("groups") or {}
    if groups:
        lines.append("shared backbones: " + ", ".join(f"{name}={gib(v):.2f} GiB" for name, v in groups.items()))
    shared_total = (stats.get("shared") or {}).get("total_bytes", 0)
    if shared_total:
        components = (stats.get("shared") or {}).get("components", {})
        lines.append(
            f"registry: {len(components)} component(s), {gib(shared_total):.2f} GiB "
            f"(saved by not duplicating: {gib(stats.get('dedup_saved_bytes', 0)):.2f} GiB)"
        )
    if smi:
        lines.append(f"nvidia-smi used {smi.get('used_mib', 0) / 1024:.2f} GiB of {smi.get('total_mib', 0) / 1024:.2f} GiB")
    return "\n".join(lines)


def markdown_section(label: str, stats: dict, smi: dict | None, env: dict, warmup: dict) -> str:
    """Render one recorded profile as a markdown section."""
    total = stats.get("total_bytes") or 0
    headroom = stats.get("headroom_bytes") or 0
    lines = [
        f"### {label}",
        "",
        f"- **When:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- **Host:** {env.get('gpu') or 'CPU'} ({gib(total):.2f} GiB) · torch {env.get('torch')} · "
        f"driver {env.get('driver') or 'n/a'} · {env.get('platform')}",
        f"- **Steady state:** torch allocated {gib(stats.get('allocated_bytes')):.2f} GiB, "
        f"reserved {gib(stats.get('reserved_bytes')):.2f} GiB, "
        f"peak {gib(stats.get('peak_allocated_bytes')):.2f} GiB"
        + (f", nvidia-smi used {smi['used_mib'] / 1024:.2f} GiB" if smi else ""),
        f"- **Device-level used:** {gib(stats.get('device_used_bytes')):.2f} GiB "
        f"(torch + CTranslate2 + CUDA contexts; source: {stats.get('device_used_source') or 'n/a'})",
        f"- **Headroom:** {gib(headroom):.2f} GiB (device free via torch.cuda.mem_get_info; "
        f"device total {gib(total):.2f} GiB)",
        f"- **Shared backbones:** {json.dumps({k: round(v / GIB, 3) for k, v in (stats.get('groups') or {}).items()})}",
        "",
        "| model | device | dtype | group | VRAM MiB | weights MiB |",
        "|---|---|---|---|---:|---:|",
    ]
    for name, detail in (stats.get("models") or {}).items():
        lines.append(
            f"| {name} | {detail.get('device') or '-'} | "
            f"{str(detail.get('dtype') or '-').replace('torch.', '')} | "
            f"{detail.get('backbone_group') or '-'} | "
            f"{gib(detail.get('vram_bytes')) * 1024:.0f} | {gib(detail.get('weights_bytes')) * 1024:.0f} |"
        )
    probes = [
        f"{name}={gib(detail.get('peak_activation_bytes')) * 1024:.0f} MiB"
        + ("" if detail.get("probe_measured") else f" (skipped: {detail.get('probe_reason')})")
        for name, detail in (stats.get("models") or {}).items()
        if detail.get("model_type") == "FasterWhisper"
    ]
    if probes:
        lines.append("")
        lines.append("Whisper max-shape probes: " + ", ".join(probes))
    if warmup:
        lines.append("")
        lines.append(
            "Warmup peaks: "
            + ", ".join(f"{family}={value}" for family, value in warmup.items())
        )
    lines.append("")
    return "\n".join(lines)


def append_markdown(path: str, section: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(section)
        if not section.endswith("\n\n"):
            handle.write("\n")


def environment() -> dict:
    info = {"torch": None, "gpu": None, "driver": None, "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            info["driver"] = proc.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover
        pass
    return info


def _warmup_family(serving_tts, family: str) -> str:
    """Run one short request for ``family`` and return a peak-memory note."""
    import torch

    from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest

    class _Request:
        is_disconnected = staticmethod(lambda: asyncio.sleep(0, result=False))

    request = TextToSpeechRequest(model=family, input=WARMUP_TEXT, mode="single_speaker_tts")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    response = asyncio.run(serving_tts.create_text_to_speech(request, _Request()))
    ok = bool(getattr(response, "data", None))
    peak = torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0.0
    return f"{'ok' if ok else 'empty'} {time.time() - started:.1f}s peak={peak:.2f}GiB"


def run(args) -> int:
    from harmonyspeech.common.config import EngineConfig
    from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
    from harmonyspeech.engine.args_tools import AsyncEngineArgs
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

    config = EngineConfig.load_config_from_yaml(args.config)
    env = environment()
    smi_before = query_nvidia_smi(args.device_index)
    boot_started = time.time()
    engine = AsyncHarmonySpeech.from_engine_args_and_config(
        AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True), config, start_engine_loop=True
    )
    boot_seconds = time.time() - boot_started
    serving_tts = OpenAIServingTextToSpeech(
        engine, OpenAIServingTextToSpeech.models_from_config(config.model_configs)
    )

    stats = collect_vram_stats(engine.engine)
    warmup: dict[str, str] = {}
    if not args.no_warmup:
        available = set(getattr(engine.engine, "available_families", []) or [])
        for family in WARMUP_FAMILIES:
            if family not in available:
                continue
            try:
                warmup[family] = _warmup_family(serving_tts, family)
            except Exception as exc:  # noqa: BLE001 - profile best effort
                warmup[family] = f"error: {type(exc).__name__}: {exc}"
        stats = collect_vram_stats(engine.engine)

    smi_after = query_nvidia_smi(args.device_index)
    stats["boot_seconds"] = round(boot_seconds, 1)

    label = args.label or os.path.basename(args.config)
    print(f"VRAM profile — {label}")
    print(format_table(stats, smi_after))
    if warmup:
        print("warmup:", json.dumps(warmup, indent=2))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "label": label,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": os.path.abspath(args.config),
            "env": env,
            "nvidia_smi_before": smi_before,
            "nvidia_smi_after": smi_after,
            "stats": stats,
            "warmup": warmup,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}")

    if args.append:
        append_markdown(args.append, markdown_section(label, stats, smi_after, env, warmup))
        print(f"appended a section to {args.append}")

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure engine VRAM footprint (Phase 4-5)")
    parser.add_argument("--config", default="config.gpu.yml")
    parser.add_argument("--label", default=None)
    parser.add_argument("--json-out", default="out/vram/latest.json")
    parser.add_argument("--append", default=None, help="append a markdown section to this file")
    parser.add_argument("--no-warmup", action="store_true", help="skip per-family warmup requests")
    parser.add_argument("--device-index", type=int, default=0)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
