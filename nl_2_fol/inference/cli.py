import os

from jsonargparse import CLI

from nl_2_fol.inference.learner.learn_definitions import LearnDefinitions
from nl_2_fol.inference.learner.validator import PerformValidation
from nl_2_fol.prompting.chebai_prompt import ChebiPrompt
from nl_2_fol.prompting.llm_inference import API_PLATFORM
from typing import Optional


# should be the root of the repo
# eg. G:\github-aditya0by0\chebai-NL2FOL\
WORKING_DIR = os.getcwd()

PROJECT_DIR = os.path.join(WORKING_DIR, "nl_2_fol")
DATA_DIR = os.path.join(WORKING_DIR, "data")
PROMPT_TEMPLATES_DIR = os.path.join(PROJECT_DIR, "prompting", "prompt_templates")


class Main:
    @staticmethod
    def learn(
        class_name: str = "all",
        api_platform: API_PLATFORM = "anthropic",
        model_name: str = "claude-opus-4-6",
        system_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "system_prompts", "with_predicates_list.yaml"
        ),
        few_shot_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "few_shots", "with_DL_style.json"
        ),
        err_failure_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "failure", "error_prompt.yaml"
        ),
        undef_failure_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "failure", "predicates_undef_with_eg.yaml"
        ),
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/slim_dataset.csv
        slim_dataset_path: str = os.path.join(DATA_DIR, "classes_slim.csv"),
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/structures.csv
        structures_data_path: str = os.path.join(DATA_DIR, "structures.csv"),
        max_attempts: int = 3,
        f1_threshold: float = 0.8,
    ):
        chebai_prompt = ChebiPrompt(
            platform=api_platform,
            model_name=model_name,
            system_prompt_fp=system_prompt_fp,
            few_shot_prompt_fp=few_shot_prompt_fp,
            err_failure_prompt_fp=err_failure_prompt_fp,
            undef_failure_prompt_fp=undef_failure_prompt_fp,
        )

        learner = LearnDefinitions(
            chebi_prompt_obj=chebai_prompt,
            slim_dataset_path=slim_dataset_path,
            structures_path=structures_data_path,
            max_attempts=max_attempts,
            f1_threshold=f1_threshold,
            fol_reasoner="gavel",
        )
        if class_name == "all":
            learner.learn_fol_definitions()
        else:
            learner.learn_class(class_name=class_name)

    @staticmethod
    def validate(
        defs_file_path: str,  # nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions.pkl
        class_name: str | None = None,
        class_names_txt_file_path: str | None = None,
        file_save_index: Optional[int] = None,
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/slim_dataset.csv
        slim_dataset_path: str = os.path.join(DATA_DIR, "classes_slim.csv"),
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/structures.csv
        structures_data_path: str = os.path.join(DATA_DIR, "structures.csv"),
    ):
        """
        Validate learned definitions.

        Parameters
        - `defs_file_path`: path to the pickled `DefinitionLearningResults` produced
            by the learning pipeline.
        - `class_name`: validate a single class by name or all classes in single job.
        - `class_names_txt_file_path`: path to a plain-text file containing class
            names (one per line). This was added so you can share validation across
            multiple HPC jobs: provide disjoint class lists to different jobs and
            validate them in parallel. Each class validation is expensive (the
            pipeline validates against ~35,700 samples), so running jobs in
            parallel for disjoint class subsets completes the overall run faster.

        The validator performs atomic, locked merges when writing the output
        pickle, so multiple jobs can safely update the same definitions file.
        The text file should contain the exact class names used in the learned
        definitions (one per line, no header).
        """

        validator = PerformValidation(
            defs_file_path=defs_file_path,
            slim_dataset_path=slim_dataset_path,
            structures_path=structures_data_path,
        )
        if class_names_txt_file_path is not None and class_name is not None:
            raise ValueError(
                "Use either class_name or class_names_txt_file_path, not both."
            )

        if class_names_txt_file_path is not None:
            if file_save_index is None:
                raise ValueError("Need file save index with classes list")

            with open(class_names_txt_file_path, "r") as f:
                class_names = [line.strip() for line in f]
            if len(class_names) == 0:
                raise ValueError("class_names_txt_file is empty.")
            validator.validate(class_names=class_names, file_save_index=int(file_save_index))
        elif class_name == "all":
            # validate all classes in the definitions file (this is a single-job, non-parallel option)
            validator.validate()
        elif class_name is not None:
            validator.validate_class(class_name=class_name)
        else:
            raise ValueError(
                "Either class_name or class_names_txt_file_path must be provided."
            )

    @staticmethod
    def learn_mistral(
        class_name: str = "all",
        api_platform: API_PLATFORM = "ollama",
        model_name: str = "my-mistral",
        system_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "system_prompts", "mistral_fol_finetuned.yaml"
        ),
        few_shot_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "few_shots", "mistral_fol_math_syntax.json"
        ),
        err_failure_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "failure", "error_prompt.yaml"
        ),
        undef_failure_prompt_fp: str = os.path.join(
            PROMPT_TEMPLATES_DIR, "failure", "mistral_fol_undefined.yaml"
        ),
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/slim_dataset.csv
        slim_dataset_path: str = os.path.join(DATA_DIR, "classes_slim.csv"),
        # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/structures.csv
        structures_data_path: str = os.path.join(DATA_DIR, "structures.csv"),
        max_attempts: int = 3,
        f1_threshold: float = 0.8,
    ):
        chebai_prompt = ChebiPrompt(
            platform=api_platform,
            model_name=model_name,
            system_prompt_fp=system_prompt_fp,
            few_shot_prompt_fp=few_shot_prompt_fp,
            err_failure_prompt_fp=err_failure_prompt_fp,
            undef_failure_prompt_fp=undef_failure_prompt_fp,
        )

        learner = LearnDefinitions(
            chebi_prompt_obj=chebai_prompt,
            slim_dataset_path=slim_dataset_path,
            structures_path=structures_data_path,
            max_attempts=max_attempts,
            f1_threshold=f1_threshold,
            fol_reasoner="mistral",
        )
        if class_name == "all":
            learner.learn_fol_definitions()
        else:
            learner.learn_class(class_name=class_name)


if __name__ == "__main__":
    # python nl_2_fol/inference/cli.py learn --help
    CLI(Main)
    # Main.validate(
    #     defs_file_path=os.path.join(
    #         PROJECT_DIR,
    #         "inference",
    #         "learner",
    #         "learned",
    #         "claude-opus-4-6",
    #         "learned_definitions_a3_with_val.pkl",
    #     )
    # )
