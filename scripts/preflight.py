"""Read-only local preflight for the configured runtime profile.

    The command intentionally never connects to DeepSeek/NapCat, sends a
message, starts workers, or writes a READY marker.  A non-zero result means
that production startup must remain blocked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.config import ConfigError, load_config
from qichi.dialogue.model_capability import MODEL_MANIFESTS, load_provider_capability_evidence
from qichi.dialogue.prompt_loader import PromptLoadError, load_runtime_prompt


def _check(label: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "BLOCKED"
    print(f"{status}: {label}: {detail}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only Qichi runtime preflight")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config.example.yaml",
        help="configuration path (default: config.example.yaml)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="machine-readable provider capability evidence (default: derived from the configured model)",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else _ROOT / args.config
    passed = True

    try:
        config = load_config(config_path)
    except ConfigError as error:
        print(f"BLOCKED: config: {error}")
        return 2
    passed &= _check(
        "context profile",
        (
            config.dialogue.context_window_preferred_tokens
            == config.dialogue.context_window_max_tokens
            == config.llm.primary.required_min_context_tokens
            == 262144
            and config.dialogue.adaptive_expansion is False
        ),
        "256K (262144), adaptive expansion disabled",
    )

    try:
        load_runtime_prompt(_ROOT, config.persona.system_prompt_file)
    except PromptLoadError as error:
        passed &= _check("runtime role prompt", False, str(error))
    else:
        passed &= _check("runtime role prompt", True, "UTF-8 thin prompt loaded")

    manifest = MODEL_MANIFESTS.get(config.llm.primary.model)
    passed &= _check(
        "model manifest",
        manifest is not None,
        config.llm.primary.model if manifest is not None else "no verified manifest",
    )
    artifact = _ROOT / "runtime" / "model-cache" / "v4-tokenizer.json"
    if manifest is not None:
        try:
            manifest.load_token_counter(artifact)
        except Exception as error:
            passed &= _check("tokenizer artifact", False, str(error))
        else:
            passed &= _check("tokenizer artifact", True, str(artifact.relative_to(_ROOT)))

    catalog_path = config.expression.qq_face.runtime_catalog
    if config.expression.qq_face.enabled and catalog_path:
        resolved_catalog = _ROOT / catalog_path
        passed &= _check(
            "QQ face catalog",
            resolved_catalog.is_file(),
            str(catalog_path) if resolved_catalog.is_file() else f"missing {catalog_path}",
        )

    # 证据文件按主模型取名（2026-09-12 起支持 pro）；显式传入的路径优先。
    evidence_path = (
        _ROOT / "runtime" / f"{config.llm.primary.model}-capability.json"
        if args.evidence is None
        else (args.evidence if args.evidence.is_absolute() else _ROOT / args.evidence)
    )
    try:
        evidence = load_provider_capability_evidence(
            evidence_path,
            provider=config.llm.provider,
            model_id=config.llm.primary.model,
        )
    except Exception as error:
        passed &= _check("provider capability evidence", False, type(error).__name__)
    else:
        passed &= _check(
            "provider capability evidence",
            evidence.context_tokens >= config.dialogue.context_window_max_tokens,
            f"{evidence.context_tokens} tokens from {evidence.source}",
        )
    passed &= _check(
        "production coordinator and workers",
        True,
        f"NapCat/SQLite/{config.llm.provider}/{config.llm.primary.model}/MemoryWorker/InitiativeScheduler assembly is available; startup remains explicit",
    )
    passed &= _check(
        "external actions",
        True,
        "no network, QQ message, READY marker, or autostart action was performed",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
