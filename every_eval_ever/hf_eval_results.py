"""Export Every Eval Ever aggregate records to Hugging Face eval YAML.

Hugging Face Community Eval results live in model repositories under
``.eval_results/*.yaml``.  This module converts EEE aggregate JSON files into
that lightweight format while linking back to the full EEE datastore record.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EEE_DATASTORE_BASE_URL = (
    'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/main'
)


@dataclass(frozen=True)
class BenchmarkMapping:
    """HF benchmark/task mapping for an EEE benchmark."""

    hf_dataset_id: str
    hf_task_id: str
    output_file: str


DEFAULT_BENCHMARK_MAPPINGS: dict[str, BenchmarkMapping] = {
    # HF docs use this as the canonical Community Evals GPQA example.
    'gpqa-diamond': BenchmarkMapping(
        hf_dataset_id='Idavidrein/gpqa',
        hf_task_id='gpqa_diamond',
        output_file='gpqa.yaml',
    ),
    'gpqa diamond': BenchmarkMapping(
        hf_dataset_id='Idavidrein/gpqa',
        hf_task_id='gpqa_diamond',
        output_file='gpqa.yaml',
    ),
    'reasoningmia/gpqa_diamond': BenchmarkMapping(
        hf_dataset_id='Idavidrein/gpqa',
        hf_task_id='gpqa_diamond',
        output_file='gpqa.yaml',
    ),
    # HF docs list GSM8K as a registered example benchmark.
    'gsm8k': BenchmarkMapping(
        hf_dataset_id='openai/gsm8k',
        hf_task_id='gsm8k',
        output_file='gsm8k.yaml',
    ),
    'openai/gsm8k': BenchmarkMapping(
        hf_dataset_id='openai/gsm8k',
        hf_task_id='gsm8k',
        output_file='gsm8k.yaml',
    ),
}


@dataclass(frozen=True)
class HFEvalResult:
    """One Hugging Face ``.eval_results`` YAML entry."""

    model_id: str
    dataset_id: str
    task_id: str
    value: int | float
    source_url: str
    source_name: str = 'EvalEval'
    date: str | None = None
    notes: str | None = None
    output_file: str = 'eval_results.yaml'


def _mapping_key(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower()


def load_mapping_json(path: Path) -> dict[str, BenchmarkMapping]:
    """Load additional benchmark mappings from JSON.

    Expected format::

        {
          "gpqa-diamond": {
            "hf_dataset_id": "Idavidrein/gpqa",
            "hf_task_id": "gpqa_diamond",
            "output_file": "gpqa.yaml"
          }
        }
    """

    with path.open(encoding='utf-8') as file:
        raw = json.load(file)

    mappings: dict[str, BenchmarkMapping] = {}
    for key, value in raw.items():
        mappings[_mapping_key(key) or key] = BenchmarkMapping(
            hf_dataset_id=value['hf_dataset_id'],
            hf_task_id=value['hf_task_id'],
            output_file=value.get('output_file', f'{key}.yaml'),
        )
    return mappings


def datastore_relative_path(input_path: Path) -> str:
    """Return ``data/...`` path used in EEE datastore backlinks."""

    parts = input_path.as_posix().split('/')
    if 'data' in parts:
        return '/'.join(parts[parts.index('data') :])
    return input_path.as_posix().lstrip('./')


def benchmark_from_path(input_path: Path) -> str | None:
    """Infer EEE benchmark directory from a datastore-style path."""

    parts = input_path.as_posix().split('/')
    if 'data' not in parts:
        return None
    data_index = parts.index('data')
    if len(parts) <= data_index + 1:
        return None
    return parts[data_index + 1]


def source_url_for_path(
    input_path: Path, base_url: str = EEE_DATASTORE_BASE_URL
) -> str:
    """Build a stable backlink to the full EEE datastore JSON record."""

    return f'{base_url.rstrip("/")}/{datastore_relative_path(input_path)}'


def normalize_date(value: Any) -> str | None:
    """Normalize EEE timestamps to the HF-friendly ``YYYY-MM-DD`` form."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # Some older EEE records store Unix timestamps as strings/floats.
    try:
        if re.fullmatch(r'\d+(\.\d+)?', text):
            return datetime.fromtimestamp(float(text), tz=UTC).date().isoformat()
    except (OverflowError, ValueError):
        pass

    # ISO-8601 datetime or date.  HF accepts datetimes, but dates make stable
    # golden tests and compact YAML.
    iso_text = text.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(iso_text).date().isoformat()
    except ValueError:
        pass

    match = re.match(r'^(\d{4}-\d{2}-\d{2})', text)
    if match:
        return match.group(1)
    return None


def _find_mapping(
    *,
    benchmark: str | None,
    evaluation_name: str | None,
    hf_repo: str | None,
    mappings: dict[str, BenchmarkMapping],
) -> BenchmarkMapping | None:
    for candidate in (benchmark, evaluation_name, hf_repo):
        key = _mapping_key(candidate)
        if key and key in mappings:
            return mappings[key]
    return None


def convert_eee_record_to_hf_results(
    input_path: Path,
    *,
    mappings: dict[str, BenchmarkMapping] | None = None,
    source_base_url: str = EEE_DATASTORE_BASE_URL,
) -> list[HFEvalResult]:
    """Convert one EEE aggregate JSON file to HF eval result entries.

    Results whose benchmark is not mapped to an HF Benchmark dataset/task are
    skipped.  Non-HF source datasets and missing scores are also skipped.
    """

    active_mappings = {**DEFAULT_BENCHMARK_MAPPINGS, **(mappings or {})}
    with input_path.open(encoding='utf-8') as file:
        record = json.load(file)

    model_id = (record.get('model_info') or {}).get('id')
    if not model_id:
        return []

    path_benchmark = benchmark_from_path(input_path)
    fallback_date = normalize_date(record.get('evaluation_timestamp'))
    source_url = source_url_for_path(input_path, source_base_url)

    results: list[HFEvalResult] = []
    for eval_result in record.get('evaluation_results') or []:
        source_data = eval_result.get('source_data') or {}
        if source_data.get('source_type') != 'hf_dataset':
            continue

        score = (eval_result.get('score_details') or {}).get('score')
        if score is None:
            continue

        mapping = _find_mapping(
            benchmark=path_benchmark,
            evaluation_name=eval_result.get('evaluation_name'),
            hf_repo=source_data.get('hf_repo'),
            mappings=active_mappings,
        )
        if mapping is None:
            continue

        results.append(
            HFEvalResult(
                model_id=model_id,
                dataset_id=mapping.hf_dataset_id,
                task_id=mapping.hf_task_id,
                value=score,
                date=normalize_date(eval_result.get('evaluation_timestamp'))
                or fallback_date,
                source_url=source_url,
                output_file=mapping.output_file,
            )
        )

    return results


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int | float):
        return str(value)
    # JSON string quoting is valid YAML and avoids PyYAML as a hard dependency.
    return json.dumps(str(value), ensure_ascii=False)


def hf_results_to_yaml(results: list[HFEvalResult]) -> str:
    """Serialize HF eval results to the YAML shape expected by the Hub."""

    lines: list[str] = []
    for result in results:
        lines.extend(
            [
                '- dataset:',
                f'    id: {_yaml_scalar(result.dataset_id)}',
                f'    task_id: {_yaml_scalar(result.task_id)}',
                f'  value: {_yaml_scalar(result.value)}',
            ]
        )
        if result.date:
            lines.append(f'  date: {_yaml_scalar(result.date)}')
        lines.extend(
            [
                '  source:',
                f'    url: {_yaml_scalar(result.source_url)}',
                f'    name: {_yaml_scalar(result.source_name)}',
            ]
        )
        if result.notes:
            lines.append(f'  notes: {_yaml_scalar(result.notes)}')
    return '\n'.join(lines) + ('\n' if lines else '')


def write_hf_eval_results(
    results: list[HFEvalResult], output_dir: Path, *, flat: bool = False
) -> list[Path]:
    """Write results grouped by model and output YAML filename.

    By default files are written under::

        {output_dir}/{model_id}/.eval_results/{output_file}

    With ``flat=True`` they are written under::

        {output_dir}/.eval_results/{output_file}
    """

    grouped: dict[tuple[str, str], list[HFEvalResult]] = {}
    for result in results:
        model_key = '' if flat else result.model_id
        grouped.setdefault((model_key, result.output_file), []).append(result)

    written: list[Path] = []
    for (model_id, output_file), grouped_results in sorted(grouped.items()):
        target_dir = output_dir / model_id / '.eval_results'
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / output_file
        target_path.write_text(hf_results_to_yaml(grouped_results), encoding='utf-8')
        written.append(target_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever hf-eval-results',
        description='Convert EEE aggregate JSON records to HF .eval_results YAML.',
    )
    parser.add_argument('paths', nargs='+', help='EEE aggregate JSON files.')
    parser.add_argument(
        '--output-dir',
        default='hf_eval_results',
        help='Directory where generated .eval_results files are written.',
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Write directly to OUTPUT_DIR/.eval_results for a single model repo.',
    )
    parser.add_argument(
        '--mapping-json',
        help='Optional JSON file with additional EEE benchmark -> HF task mappings.',
    )
    parser.add_argument(
        '--source-base-url',
        default=EEE_DATASTORE_BASE_URL,
        help='Base URL for EEE datastore backlinks.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    extra_mappings = (
        load_mapping_json(Path(args.mapping_json)) if args.mapping_json else {}
    )
    all_results: list[HFEvalResult] = []
    for path_str in args.paths:
        all_results.extend(
            convert_eee_record_to_hf_results(
                Path(path_str),
                mappings=extra_mappings,
                source_base_url=args.source_base_url,
            )
        )

    written = write_hf_eval_results(
        all_results, Path(args.output_dir), flat=args.flat
    )
    for path in written:
        print(path)
    print(f'Wrote {len(written)} file(s) with {len(all_results)} result(s).')
    return 0
