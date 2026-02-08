import unittest

from nl_2_fol.prompting.few_shot import ChebaiFewShotPrompt, CHEBIFOLOutput


class TestChebaiFewShotPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fs_obj = ChebaiFewShotPrompt(
            platform="groq",
            model_name="openai/gpt-oss-120b",
            system_prompt_fp="configs/system_prompt.yaml",
            few_shot_prompt_fp="configs/few_shots_prompts.json",
        )
        cls._fs_obj.print_whole_prompt_with_given_input(
            """CHEBI:16236 - ethanol: A primary alcohol that
            is ethane in which one of the hydrogens is substituted
            by a hydroxy group."""
        )

    def print_result(self, result: CHEBIFOLOutput) -> None:
        print("Relevant Definition: ", result.intermediate_output.relevant_definition)
        print("Superclass: ", result.intermediate_output.superclass)
        print("Explanation: ", result.intermediate_output.explanation)
        print()
        print("FOL Formula: ", result.FOL_formula)

    def test_invoke_llm_with_17245(self) -> None:
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:17245 - carbon monoxide: A one-carbon compound in which
            the carbon is joined only to a single oxygen.
            It is a colourless, odourless, tasteless, toxic gas."""
        )
        self.print_result(result)

    def test_invoke_llm_with_17790(self) -> None:
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:17790 - methanol: The primary alcohol that
            is the simplest aliphatic alcohol, comprising a methyl and an alcohol group."""
        )
        self.print_result(result)

    def test_invoke_llm_with_16526(self) -> None:
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:16526 - carbon dioxide: A one-carbon compound with formula CO2
            in which the carbon is attached to each oxygen atom by a double bond.
            A colourless, odourless gas under normal conditions, it is produced
            during respiration by all animals, fungi and microorganisms that depend
            directly or indirectly on living or decaying plants for food."""
        )
        self.print_result(result)

    def test_invoke_llm_with_16134(self) -> None:
        """Test case for CHEBI:16134 - Ammonia"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:16134 - ammonia: An azane that consists of a single nitrogen atom covelently bonded to three hydrogen atoms."""
        )
        self.print_result(result)

    def test_invoke_llm_with_16183(self) -> None:
        """Test case for CHEBI:16183 - Methane"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:16183 - methane: A one-carbon compound in which the carbon
            is attached by single bonds to four hydrogen atoms. It is a colourless,
            odourless, non-toxic but flammable gas (b.p. −161°C)."""
        )
        self.print_result(result)

    def test_invoke_llm_with_26710(self) -> None:
        """Test case for CHEBI:26710 - Sodium Chloride"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:26710 - sodium chloride: An inorganic chloride salt having
            sodium(1+) as the counterion."""
        )
        self.print_result(result)

    def test_invoke_llm_with_30751(self) -> None:
        """Test case for CHEBI:30751 - Formic Acid"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:30751 - formic acid: The simplest carboxylic acid,
            containing a single carbon. Occurs naturally in various sources
            including the venom of bee and ant stings, and is a useful organic
            synthetic reagent. Principally used as a preservative and antibacterial
            agent in livestock feed. Induces severe metabolic acidosis and ocular
            injury in human subjects."""
        )
        self.print_result(result)

    def test_invoke_llm_with_15366(self) -> None:
        """Test case for CHEBI:15366 - Acetic Acid"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:15366 - acetic acid: A simple monocarboxylic acid containing
            two carbons."""
        )
        self.print_result(result)

    def test_invoke_llm_with_27732(self) -> None:
        """Test case for CHEBI:27732 - Caffeine"""
        result = self._fs_obj.invoke_llm_with_fs_prompt(
            """CHEBI:27732 - caffeine: A trimethylxanthine in which the three methyl
            groups are located at positions 1, 3, and 7. A purine alkaloid that occurs
            naturally in tea and coffee."""
        )
        self.print_result(result)


if __name__ == "__main__":
    unittest.main()
