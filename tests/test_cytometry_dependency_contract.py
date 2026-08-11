"""Packaging contracts for the optional cytometry reader."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_every_cytometry_extra_requires_flowio_with_as_array():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    optional = project["project"]["optional-dependencies"]

    for extra in ("cytometry", "cytometry-full"):
        flowio = next(dep for dep in optional[extra] if dep.startswith("flowio"))
        assert flowio == "flowio>=1.4", (
            f"{extra} must require flowio>=1.4 because read_fcs calls "
            "FlowData.as_array(), which is absent from flowio 1.3"
        )
