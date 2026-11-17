"""Prerequisites for checks that replay locally reconstructed or saved artifacts."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


@pytest.fixture
def require_local_files():
    def require(paths, *, reason):
        missing = [Path(path) for path in paths if not Path(path).is_file()]
        if missing:
            names = ', '.join(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                              else str(path) for path in missing[:3])
            pytest.skip(f'{reason}; missing {names}. See README: Tests.')
    return require


@pytest.fixture
def require_training_inputs(require_local_files):
    def require(config):
        if config.get('granola_data_dir'):
            directory = ROOT/config['granola_data_dir']
            require_local_files(
                [directory/name for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl')],
                reason='Reconstruct the Granola splits with data_preparation.granola.restore')
        directory = ROOT/config['data_dir']
        if ROOT/'ablations/lfm/greedy-control' in directory.parents:
            require_local_files(
                [directory/name for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl',
                                              'case_targets.jsonl', 'manifest.json')],
                reason='Rebuild the matched-control datasets with data_preparation.vital_persons.greedy_control')
    return require
