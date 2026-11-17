"""The penalty comparison must reproduce the primary full-hierarchy utility."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_error_minus_one_exactly_matches_the_main_evaluation(require_local_files):
    report_path = ROOT/'results-metric-sensitivity/test/summary.json'
    require_local_files([report_path], reason='Saved metric-sensitivity reports are local-only')
    report = json.loads(report_path.read_text())
    require_local_files(
        [ROOT/'results-corrected'/model/'test/summary.json' for model in report['models']],
        reason='Saved primary evaluation summaries are local-only')
    assert {m['error_score'] for m in report['metrics']} == {-1, 0}
    for model, blocks in report['models'].items():
        source = json.loads((ROOT/'results-corrected'/model/'test/summary.json').read_text())
        for domain, block in blocks.items():
            for method, record in block['methods'].items():
                headline = source['modalities'][domain]['variants'][method]['headline']
                assert record['accuracy'] == headline['truth_accuracy']
                assert record['scores']['ig_penalty'] == headline['uniform_error_utility']
                assert record['scores']['ig_zero'] == headline['positive_information_reward']


def test_case_counts_are_identical_across_methods(require_local_files):
    report_path = ROOT/'results-metric-sensitivity/test/summary.json'
    require_local_files([report_path], reason='Saved metric-sensitivity reports are local-only')
    report = json.loads(report_path.read_text())
    for blocks in report['models'].values():
        for block in blocks.values():
            assert len({r['n'] for r in block['methods'].values()}) == 1
