"""Sample Qichi's synthetic quality baseline through the primary model only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qichi.config import load_config
from qichi.dialogue.model_capability import MODEL_MANIFESTS
from qichi.dialogue.prompt_loader import load_runtime_prompt
from qichi.expression.catalog import load_qq_expression_catalog
from qichi.quality import (
    build_quality_sampler,
    load_scenarios_jsonl,
    sample_scenarios_to_jsonl,
)


def _hydrate_user_environment() -> None:
    if os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in (
                "QICHI_OWNER_QQ",
                "NAPCAT_WS_URL",
                "NAPCAT_HTTP_URL",
                "NAPCAT_ACCESS_TOKEN",
                "SILICONFLOW_API_KEY",
                "DEEPSEEK_API_KEY",
            ):
                if name in os.environ:
                    continue
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if isinstance(value, str) and value:
                    os.environ[name] = value
    except (ImportError, OSError):
        return


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", type=Path, help="existing quality-baseline JSONL")
    parser.add_argument("output", type=Path, help="destination JSONL for model samples")
    parser.add_argument("--config", type=Path, default=ROOT / "config.example.yaml")
    parser.add_argument("--samples-per-scenario", type=int, default=3)
    return parser.parse_args(argv)


def _face_keys(config: object) -> tuple[str, ...]:
    qq_face = config.expression.qq_face
    if not qq_face.enabled:
        return ()
    if not qq_face.runtime_catalog:
        raise ValueError("enabled QQ face catalog is unavailable")
    path = Path(qq_face.runtime_catalog)
    catalog = load_qq_expression_catalog(path if path.is_absolute() else ROOT / path)
    return tuple(catalog.faces)


async def _run(args: argparse.Namespace) -> int:
    _hydrate_user_environment()
    config = load_config(args.config)
    scenarios = load_scenarios_jsonl(args.scenarios)
    role_prompt = load_runtime_prompt(ROOT, config.persona.system_prompt_file)
    manifest = MODEL_MANIFESTS.get(config.llm.primary.model)
    if manifest is None:
        raise ValueError("configured model has no verified manifest")
    token_counter = manifest.load_token_counter(ROOT / "runtime" / "model-cache" / "v4-tokenizer.json")
    if config.expression.message_reaction.enabled:
        raise ValueError("enabled reaction catalog is unavailable")
    sampler = build_quality_sampler(
        config,
        token_counter,
        face_keys=_face_keys(config),
        reaction_keys=(),
    )
    try:
        records = await sample_scenarios_to_jsonl(
            scenarios,
            role_prompt=role_prompt,
            sampler=sampler,
            output_path=args.output,
            samples_per_scenario=args.samples_per_scenario,
            source_path=args.scenarios,
        )
    finally:
        await sampler.close()
    counts = {kind: sum(item.result == kind for item in records) for kind in ("reply", "skip", "failure")}
    print(
        json.dumps(
            {"output": str(args.output), "sample_count": len(records), "results": counts},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(_args(argv)))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"quality_sample: failed ({type(error).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
