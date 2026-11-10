"""CUDA-only kernels never enter the CPU/XPU base closure."""

import tomllib
from pathlib import Path


def test_cuda_dependencies_are_not_required() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
    assert not any(requirement.startswith(("cuequivariance", "transformer-engine")) for requirement in project["dependencies"])


if __name__ == "__main__":
    test_cuda_dependencies_are_not_required()
