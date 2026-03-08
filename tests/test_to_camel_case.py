import pytest

from nl_2_fol.inference.utils.to_camel_case import to_camel_case


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("epoxy fatty acid", "epoxyFattyAcid"),
        ("EPOXY FATTY ACID", "epoxyFattyAcid"),
        ("  epoxy   fatty-acid  ", "epoxyFattyAcid"),
        ("omega 3 fatty acid", "omega3FattyAcid"),
        ("class_1 example", "class1Example"),
        ("short-chain fatty acyl-CoA", "shortChainFattyAcylCoa"),
        ('"2,5-diketopiperazines"', "2,5Diketopiperazines"),
        ('"1,2,4-triazines"', "1,2,4Triazines"),
        ("3-substituted propionyl-CoA(4-)", "3SubstitutedPropionylCoa4"),
        (
            "N-acyl-1-O-beta-D-glucosyl-15-methylhexadecasphing-4-enine",
            "nAcyl1OBetaDGlucosyl15Methylhexadecasphing4Enine",
        ),
        ("B vitamin", "bVitamin"),
        ("CDP-diacylglycerol", "cdpDiacylglycerol"),
        ("16beta-hydroxy steroids", "16betaHydroxySteroids"),
        ("3beta-hydroxy-Delta(5)-steroids", "3betaHydroxyDelta5Steroids"),
        ("4'-hydroxyflavanones", "4Hydroxyflavanones"),
        (
            "2'-deoxyribonucleoside 5'-monophosphate",
            "2Deoxyribonucleoside5Monophosphate",
        ),
        ("sphingomyelin d18:1", "sphingomyelinD18:1"),
        ('"11,12-saturated fatty acyl-CoA(4-)"', "11,12SaturatedFattyAcylCoa4"),
        ("O-acyl-L-carnitine", "oAcylLCarnitine"),
        ("nucleoside 5'-phosphate", "nucleoside5Phosphate"),
        ("1-phosphatidyl-1D-myo-inositol", "1Phosphatidyl1DMyoInositol"),
        ("(2S)-flavan-4-one", "2SFlavan4One"),
        ("3-sn-phosphatidyl-L-serine", "3SnPhosphatidylLSerine"),
        ("3-oxo-5alpha-steroid", "3Oxo5alphaSteroid"),
        ("UDP-sugar", "udpSugar"),
        ("HETE", "hete"),
        ("para-terphenyl", "paraTerphenyl"),
        ("3-oxo-Delta(1) steroid", "3OxoDelta1Steroid"),
        ("L-alpha-amino acid", "lAlphaAminoAcid"),
        ("17alpha-hydroxy steroid", "17alphaHydroxySteroid"),
        ("vitamin D", "vitaminD"),
        ("", ""),
        ("---___   ", ""),
    ],
)
def test_to_camel_case(raw_name: str, expected: str):
    assert to_camel_case(raw_name) == expected
