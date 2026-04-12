from lark import Lark, UnexpectedCharacters

from cfg.ast import FOLTransformer, Node
from cfg.naming import NamingError


class CFGParser:
    def __init__(self):
        with open("cfg/syntax.lark", "r") as file:
            grammar = file.read()
        self.parser = Lark(grammar, parser="earley")

    def parse(self, text: str) -> Node:
        try:
            tree = self.parser.parse(text)
            return FOLTransformer().transform(tree)
        except UnexpectedCharacters as e:
            raise NamingError(self.parser, e, text)
