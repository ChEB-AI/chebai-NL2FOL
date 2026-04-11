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

from .cfgparser import CFGParser

__all__ = ["CFGParser"]
