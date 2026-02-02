from rdkit import Chem


def mol_to_fol(mol: Chem.Mol):
    """Convert an RDKit molecule to a first-order logic representation."""
    from chemlog.fol_classification.mol_to_fol_atoms import mol_to_fol_atoms
    universe, extensions = mol_to_fol_atoms(mol)

    # rename / add custom extensions if needed

    return universe, extensions
