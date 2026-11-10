"""CUDA kernels belong to explicit extras, never the CPU/XPU base closure."""

import tomllib
from pathlib import Path


def test_cuda_dependencies_are_optional() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
    assert not any(requirement.startswith("cuequivariance") for requirement in project["dependencies"])
    extras = project["optional-dependencies"]
    for cuda in ("12", "13"):
        assert "cuequivariance-torch>=0.8.1" in extras[f"cueq{cuda}"]
        assert f"cuequivariance-ops-torch-cu{cuda}>=0.8.1" in extras[f"cueq{cuda}"]


if __name__ == "__main__":
    test_cuda_dependencies_are_optional()
