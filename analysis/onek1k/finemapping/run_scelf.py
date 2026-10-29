"""Command-line entry point for scelf fine-mapping.

Run from the repository root: python -m analysis.onek1k.finemapping.run_scelf --help
"""
from ._workflow import main_for


def main(argv=None):
    main_for('scelf', argv)


if __name__ == "__main__":
    main()
