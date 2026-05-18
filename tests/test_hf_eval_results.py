from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.hf_eval_results import (
    convert_eee_record_to_hf_results,
    hf_results_to_yaml,
    normalize_date,
    source_url_for_path,
    write_hf_eval_results,
)


def _write_record(
    path: Path,
    *,
    benchmark: str = 'GPQA Diamond',
    hf_repo: str = 'reasoningMIA/gpqa_diamond',
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'schema_version': '0.2.2',
                'evaluation_id': 'gpqa-diamond/google_gemma-2-27b-it/test',
                'evaluation_timestamp': '2026-04-16T18:42:48Z',
                'retrieved_timestamp': '2026-04-17T00:00:00Z',
                'source_metadata': {
                    'source_type': 'evaluation_run',
                    'source_organization_name': 'EvalEval',
                    'evaluator_relationship': 'third_party',
                },
                'eval_library': {'name': 'test'},
                'model_info': {
                    'name': 'google/gemma-2-27b-it',
                    'id': 'google/gemma-2-27b-it',
                },
                'evaluation_results': [
                    {
                        'evaluation_name': benchmark,
                        'source_data': {
                            'dataset_name': benchmark,
                            'source_type': 'hf_dataset',
                            'hf_repo': hf_repo,
                        },
                        'metric_config': {'lower_is_better': False},
                        'score_details': {'score': 0.4090909090909091},
                        'evaluation_timestamp': '2026-04-16T18:42:48Z',
                    }
                ],
            }
        ),
        encoding='utf-8',
    )


def test_normalize_date_iso_and_unix_timestamp() -> None:
    assert normalize_date('2026-04-16T18:42:48Z') == '2026-04-16'
    assert normalize_date('1744626664.0') == '2025-04-14'


def test_source_url_uses_datastore_data_path() -> None:
    path = Path(
        '/tmp/EEE_datastore/data/gpqa-diamond/google/gemma-2-27b-it/id.json'
    )
    assert source_url_for_path(path).endswith(
        '/data/gpqa-diamond/google/gemma-2-27b-it/id.json'
    )


def test_convert_gpqa_record_to_hf_yaml(tmp_path: Path) -> None:
    record_path = tmp_path / 'data/gpqa-diamond/google/gemma-2-27b-it/id.json'
    _write_record(record_path)

    results = convert_eee_record_to_hf_results(record_path)

    assert len(results) == 1
    result = results[0]
    assert result.model_id == 'google/gemma-2-27b-it'
    assert result.dataset_id == 'Idavidrein/gpqa'
    assert result.task_id == 'gpqa_diamond'
    assert result.value == 0.4090909090909091
    assert result.date == '2026-04-16'
    assert result.output_file == 'gpqa.yaml'

    yaml_text = hf_results_to_yaml(results)
    assert '- dataset:' in yaml_text
    assert '    id: "Idavidrein/gpqa"' in yaml_text
    assert '    task_id: "gpqa_diamond"' in yaml_text
    assert '  value: 0.4090909090909091' in yaml_text
    assert '  date: "2026-04-16"' in yaml_text
    assert '    name: "EvalEval"' in yaml_text


def test_unknown_benchmark_is_skipped(tmp_path: Path) -> None:
    record_path = tmp_path / 'data/unknown-benchmark/org/model/id.json'
    _write_record(
        record_path,
        benchmark='Unknown Benchmark',
        hf_repo='example/unknown_benchmark',
    )

    assert convert_eee_record_to_hf_results(record_path) == []


def test_write_results_groups_by_model_and_eval_results_dir(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / 'data/gpqa-diamond/google/gemma-2-27b-it/id.json'
    _write_record(record_path)
    results = convert_eee_record_to_hf_results(record_path)

    written = write_hf_eval_results(results, tmp_path / 'out')

    assert written == [
        tmp_path
        / 'out/google/gemma-2-27b-it/.eval_results/gpqa.yaml'
    ]
    assert written[0].read_text(encoding='utf-8').startswith('- dataset:')
