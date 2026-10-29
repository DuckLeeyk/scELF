"""Command-line entry point for susie fine-mapping.

Run from the repository root: python -m analysis.onek1k.finemapping.run_susie --help
"""
from ._workflow import main_for


def main(argv=None):
    main_for('susie', argv)


if __name__ == "__main__":
    main()
