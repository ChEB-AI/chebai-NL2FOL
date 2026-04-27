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

from pathlib import Path

from lark import Lark, UnexpectedCharacters

from .ast import FOLTransformer, Node
from .naming import NamingError


class CFGParser:
    def __init__(self):
        grammar_path = Path(__file__).resolve().parent / "syntax.lark"
        with grammar_path.open("r", encoding="utf-8") as file:
            grammar = file.read()
        self.parser = Lark(grammar, parser="earley", lexer="standard")

    def parse(self, text: str) -> Node:
        try:
            tree = self.parser.parse(text)
            return FOLTransformer().transform(tree)
        except UnexpectedCharacters as e:
            raise NamingError(self.parser, e, text)
