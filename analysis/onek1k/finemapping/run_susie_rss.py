"""Command-line entry point for saigeqtl_susie fine-mapping.

Run from the repository root: python -m analysis.onek1k.finemapping.run_susie_rss --help
"""
from ._workflow import main_for


def main(argv=None):
    main_for('saigeqtl_susie', argv)


if __name__ == "__main__":
    main()
