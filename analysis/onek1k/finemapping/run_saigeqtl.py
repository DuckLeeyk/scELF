"""Command-line entry point for saigeqtl fine-mapping.

Run from the repository root: python -m analysis.onek1k.finemapping.run_saigeqtl --help
"""
from ._workflow import main_for


def main(argv=None):
    main_for('saigeqtl', argv)


if __name__ == "__main__":
    main()
