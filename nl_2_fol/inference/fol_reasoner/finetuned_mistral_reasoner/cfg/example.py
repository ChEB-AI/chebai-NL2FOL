"""
Code written by https://github.com/fvossel.

@misc{vossel2025advancingnaturallanguageformalization,
    title={Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs},
    author={Felix Vossel and Till Mossakowski and Bj"orn Gehrke},
    year={2025},
    eprint={2509.22338},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.22338},
}
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    # Support running this file directly: python /path/to/cfg/example.py
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from cfg.cfgparser import CFGParser

parser = CFGParser()

ast_tree = parser.parse(
    "∀x (Book(x) → KnowledgeSource(x)) ∧ ¬∀x (KnowledgeSource(x) → Book(x))"
)

print(ast_tree.to_tptp())
