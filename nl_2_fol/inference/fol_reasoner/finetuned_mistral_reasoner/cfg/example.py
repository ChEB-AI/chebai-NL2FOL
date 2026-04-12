from cfg.cfgparser import CFGParser

parser = CFGParser()

ast_tree = parser.parse(
    "∀x (Book(x) → KnowledgeSource(x)) ∧ ¬∀x (KnowledgeSource(x) → Book(x))"
)

print(ast_tree.to_tptp())
