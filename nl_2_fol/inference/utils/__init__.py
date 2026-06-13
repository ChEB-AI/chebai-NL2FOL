from nl_2_fol.inference.learner.definition_model import (
    DefinitionMetrics,
)


def print_metrics(metrics: DefinitionMetrics):
    return_str = ""
    for metric_name, value in metrics.__dict__.items():
        if metric_name not in ("fp_smiles", "fn_smiles", "timeout_smiles"):
            return_str += f"{metric_name}: {value} "
    return return_str
