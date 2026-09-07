from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_runtime_templates_and_static_assets(tmp_path: Path) -> None:
    build_root = tmp_path / "wheel-source"
    build_root.mkdir()
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", build_root / "pyproject.toml")
    shutil.copytree(PROJECT_ROOT / "src", build_root / "src")
    output_dir = tmp_path / "wheel-output"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ],
        cwd=build_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(output_dir.glob("ainovel-*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel:
        packaged_files = set(wheel.namelist())

    assert "ainovel/static/app.js" in packaged_files
    assert "ainovel/web/templates/base.html" in packaged_files
    assert "ainovel/web/templates/index.html" in packaged_files
    assert "ainovel/web/templates/project.html" in packaged_files
    assert "ainovel/web/templates/workflow.html" in packaged_files
