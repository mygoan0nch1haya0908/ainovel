import subprocess
import sys


def test_preview_cli_requires_existing_database_and_never_creates_it(tmp_path):
    missing = tmp_path / 'missing.db'
    result = subprocess.run([sys.executable, 'scripts/preview_scoped_context.py', '--database', str(missing), '--workflow', 'missing'], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 2
    assert 'database does not exist' in result.stderr
    assert not missing.exists()
