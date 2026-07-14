import pytest

from nl_2_fol.inference.utils.to_camel_case import to_camel_case


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("15G256ν", "15G256ν"),
        ("Albogrisin B′", "albogrisinB′"),
        ("Alterochromide B'", "alterochromideB'"),
        ("Alterochromide A''", "alterochromideA''"),
        ("Ganoderic acid Σ", "ganodericAcidΣ"),
        ("epoxy fatty acid", "epoxyFattyAcid"),
        ("  epoxy   fatty-acid  ", "epoxyFattyAcid"),
        ("omega 3 fatty acid", "omega3FattyAcid"),
        ("class_1 example", "class_1Example"),
        ('"2,5-diketopiperazines"', "2,5Diketopiperazines"),
        ("3-substituted propionyl-CoA(4-)", "3SubstitutedPropionylCoa(4-)"),
        (
            "N-acyl-1-O-beta-D-glucosyl-15-methylhexadecasphing-4-enine",
            "nAcyl1OBetaDGlucosyl15Methylhexadecasphing4Enine",
        ),
        ("L-alpha-amino acid", "lAlphaAminoAcid"),
        ("CDP-diacylglycerol", "cdpDiacylglycerol"),
        ("16beta-hydroxy steroids", "16betaHydroxySteroids"),
        ("3beta-hydroxy-Delta(5)-steroids", "3betaHydroxyDelta(5)Steroids"),
        (
            "2'-deoxyribonucleoside 5'-monophosphate",
            "2'Deoxyribonucleoside5'Monophosphate",
        ),
        ("sphingomyelin d18:1", "sphingomyelinD18:1"),
        ('"11,12-saturated fatty acyl-CoA(4-)"', "11,12SaturatedFattyAcylCoa(4-)"),
        ("beta-[D]-glucose", "beta[D]Glucose"),
        ("ion[Cu2+]-complex(3*/?)", "ion[Cu2+]Complex(3*/?)"),
        ("(2S)-flavan-4-one", "(2S)Flavan4One"),
        ("HETE", "hete"),
        ("NAD+", "nad+"),
        ("", ""),
        ("---___   ", "___"),
    ],
)
def test_to_camel_case(raw_name: str, expected: str):
    assert to_camel_case(raw_name) == expected
