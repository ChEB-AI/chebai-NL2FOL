from pathlib import Path

from lark import Lark, UnexpectedCharacters

from cfg.ast import FOLTransformer, Node
from cfg.naming import NamingError


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
