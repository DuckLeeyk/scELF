"""Command-line entry point for logsusie fine-mapping.

Run from the repository root: python -m analysis.onek1k.finemapping.run_logsusie --help
"""
from ._workflow import main_for


def main(argv=None):
    main_for('logsusie', argv)


if __name__ == "__main__":
    main()
