from rdkit.Chem import GetPeriodicTable


# ATOM_PREDICATES = {"c": "carbon", "o": "oxygen", ...}
# Generate atom predicates from RDKit periodic table
def get_atom_predicates() -> dict[str, str]:
    pt = GetPeriodicTable()
    atom_predicates = {}
    # Iterate through all known elements (1-118)
    for atomic_num in range(1, 119):
        try:
            symbol = pt.GetElementSymbol(atomic_num)
            name = pt.GetElementName(atomic_num)
            if symbol and name:  # Ensure valid element
                atom_predicates[symbol.lower()] = name.lower()
        except Exception:
            # Skip if element doesn't exist in this version of RDKit
            continue
    return atom_predicates


# Hydrogen atoms are not represented explicitly, but as a property of their neighbours
#     `has_{n}_hs` - exactly n hydrogens (e.g., has_0_hs, has_1_hs)
#     `has_at_least_{n}_hs` - at least n hydrogens
#     `has_min_{n}_hs` - minimum hydrogens
HYDROGEN_PREDICATES = {f"has_{n}_hs": f"has exactly {n} hydrogens" for n in range(0, 5)}

# Atom charges:
# - `charge1`, `charge_m2` for specific charges
# - `charge_p`, `charge_n` for positive or negative charge of arbitrary magnitude
# -  `charge0` for neutral charge
CHARGE_PREDICATES = {
    **{
        "charge0": "neutral charge",
        "charge_p": "positive charge of arbitrary magnitude",
        "charge_n": "negative charge of arbitrary magnitude",
    },
    # charge-3, charge-2, charge-1, charge1, charge2, charge3
    **{f"charge{n}": f"charge of {n}" for n in range(-3, 4) if n != 0},
    **{f"charge_m{-n}": f"charge of {n}" for n in range(1, 4)},
}


# Stereochemistry/Chirality (CIP codes):
# - `cip_code_R`, `cip_code_S`
CHIRALITY_PREDICATES = {
    "cip_code_R": "CIP code R (rectus) for chiral centers",
    "cip_code_S": "CIP code S (sinister) for chiral centers",
}


# Bonds:
# - `has_bond_to` (bond type unspecified)
# - `bSINGLE`, `bDOUBLE`, `bTRIPLE`, `bAROMATIC` for specific bond types
BOND_PREDICATES = {
    "has_bond_to": "has a bond to another atom (bond type unspecified)",
    "bSINGLE": "has a single bond to another atom",
    "bDOUBLE": "has a double bond to another atom",
    "bTRIPLE": "has a triple bond to another atom",
    "bAROMATIC": "has an aromatic bond to another atom",
}

# Net molecular charge:
#  - `net_charge_positive`
#  - `net_charge_negative`
#  - `net_charge_neutral`
NET_CHARGE_PREDICATES = {
    "net_charge_positive": "net positive charge for the molecule",
    "net_charge_negative": "net negative charge for the molecule",
    "net_charge_neutral": "net neutral charge for the molecule",
}

GAVEL_PREDICATES = {
    **get_atom_predicates(),
    **HYDROGEN_PREDICATES,
    **CHARGE_PREDICATES,
    **CHIRALITY_PREDICATES,
    **BOND_PREDICATES,
    **NET_CHARGE_PREDICATES,
}

__all__ = ["GAVEL_PREDICATES"]
