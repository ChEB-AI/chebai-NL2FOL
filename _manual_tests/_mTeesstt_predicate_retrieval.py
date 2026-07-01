import os

from nl_2_fol.prompting.retrieve_relevant_predicates import (
    SemanticPredicateRetriever,
)

with open(os.path.join("_manual_tests", "predicates_list.txt"), "r") as f:
    predicates = [line.strip() for line in f.readlines()]


retriever = SemanticPredicateRetriever()
retriever.add_predicates(predicates)


definition = """
CHEBI:33575 - carboxylic acid : A carbon oxoacid acid carrying at least one ‒C(=O)OH group and having the structure RC(=O)OH, where R is any any monovalent functional group. Carboxylic acids are the most common type of organic acid.

Outgoing Relation(s)

carboxylic acid (CHEBI:33575) has part carboxy group (CHEBI:46883)

carboxylic acid (CHEBI:33575) is a carbon oxoacid (CHEBI:35605)

carboxylic acid (CHEBI:33575) is a carbonyl compound (CHEBI:36586)

carboxylic acid (CHEBI:33575) is a organic acid (CHEBI:64709)

carboxylic acid (CHEBI:33575) is conjugate acid of carboxylic acid anion (CHEBI:29067)
"""

matches = retriever.retrieve_relevant_predicates(definition, top_k=None, threshold=0.35)
print(matches)
