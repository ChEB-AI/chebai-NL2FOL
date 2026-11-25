# chebai-NL2FOL

AI workflow for natural language to First-Order Logic (FOL) translation for ChEBI.


## Publication
A paper with experiments based on this repository is available on arxiv:
> (todo)


**Abstract:**

Chemical ontologies such as the Chemical Entities of Biological Interest (ChEBI) provide structured taxonomies for organizing chemical knowledge, but manually extending these taxonomies to new molecules remains a scalability challenge. We propose **ChEBI2Logic**, a pipeline for automatically translating natural language definitions of chemical classes in ChEBI into formal logic for rule-based and explainable chemical classification. ChEBI2Logic uses multi-shot LLM prompts with multiple feedback loops to generate and automatically validate logical definitions. As logical formalisms, ChEBI2Logic uses first-order logic (FOL) with TPTP syntax or ASP with gringo syntax.

The result is a classification system that is by design modular (auxiliary predicates are automatically generated and shared across definitions) and platform-independent (the definitions come in a standard FOL syntax which can be parsed by various tools).
Besides classification, the definitions can also be used to verify ontological relations directly and identify new relations.

We evaluate ChEBI2Logic on 367 ChEBI classes using different LLMs. Among these models, Claude Opus 5.5 with gringo syntax achieves the strongest performance. In comparison to C3PO, a suite of LLM-generated Python classifiers, ChEBI2Logic achieves a higher F1 score (+17.4 pp macro F1 comparing the best individual runs, +5.4 pp comparing the ensembles).
Our analysis identifies challenges for logic-based chemical classification. Overall, these results demonstrate the feasibility of natural language to logic translation for chemical classification.

<img width="2796" height="1024" alt="fig_landscape" src="https://github.com/user-attachments/assets/58dcf948-4645-4523-bb02-6120306063a0" />


## Quick Start

The default setup learns definitions as logic programs (answer set programming, evaluated with
[clingo](https://potassco.org/clingo/)) with Claude Opus 5.5 for all 367 classes of the
[C3PO](https://github.com/chemkg/c3p) dataset.

Requirements: Python >= 3.12, Linux (or WSL; model checking with time limits relies on `fork` and
`fcntl`), and a Claude subscription.

1. **Install:**

   ```bash
   pip install -e ".[dev,claude-cli]"
   ```

2. **Download the data:** put `classes_slim.csv` and `structures.csv` from the
   [C3PO dataset](https://huggingface.co/datasets/MonarchInit/C3PO/tree/main) into `data/`. ChEBI
   (version 244) is downloaded automatically to `data/chebi_v244/` on the first run.

3. **Log in to Claude:** install [Claude Code](https://code.claude.com), run `claude` and log in
   with `/login`. Claude is called through the local `claude` CLI, which bills the subscription; no
   API key is needed.

4. **Learn definitions** (from the repository root, so that the default paths resolve):

   ```bash
   python nl_2_fol/inference/cli.py learn
   ```

   Each class gets up to 3 attempts; a definition is accepted with an F1 score of at least 0.8 on
   the training samples. The definitions are saved to
   `nl_2_fol/inference/learner/learned/claude-opus-5-5_asp/learned_definitions_a3.pkl`. An
   interrupted run can be resumed by running the same command again (learned classes are
   skipped).

5. **Validate** the learned definitions on the validation set:

   ```bash
   python nl_2_fol/inference/cli.py validate \
     --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-5-5_asp/learned_definitions_a3.pkl" \
     --class_name "all"
   ```

Use `--help` on `learn`, `validate` and `learn_mistral` for all options.


## Learning

```bash
python nl_2_fol/inference/cli.py learn \
  --class_name "ethanol" \
  --api_platform "claude_cli" \
  --model_name "claude-haiku-5-5" \
  --fol_reasoner "asp" \
  --effort "high" \
  --max_attempts 3 \
  --f1_threshold 0.8
```

Definitions are saved to
`nl_2_fol/inference/learner/learned/<model_name>[_asp]/learned_definitions_a<max_attempts>.pkl`,
together with the learning log and (for Claude) the token usage per query (`llm_usage.jsonl`).

**LLM platforms** (`--api_platform`):
- `claude_cli` (default): the logged-in `claude` CLI. Set `NL2FOL_CLAUDE_CLI` if the binary is not
  on `PATH`. When the 5-hour usage window is at least 90% used (`NL2FOL_SESSION_LIMIT`), learning
  pauses until it resets; when a weekly limit is reached, the run stops and can be resumed later.
  Requests refused by Claude's safety classifiers fail the respective class only.
- `anthropic`: the Anthropic API. Install `.[anthropic]` and put `ANTHROPIC_API_KEY=...` in
  `api_keys.env` in the repository root.
- `ollama`: a local model, see [below](#running-a-custom-model-with-ollama-on-a-cluster).

**Reasoners** (`--fol_reasoner`):
- `asp` (default): definitions are logic programs, evaluated with clingo on all training samples.
  Molecules are evaluated in batches of 1,000 by parallel clingo processes (one per CPU); each batch
  has a time limit of 300 seconds. Molecules in batches that exceed it are counted as timeouts
  (`timeout_pos` / `timeout_neg` in the metrics), not as negative predictions.
- `gavel`: the TPTP pipeline of the first version of the paper (model checking with Gavel /
  ChemLog), using the positive samples and the 1,000 closest negative samples with a time limit of
  10 minutes.

The default prompt templates depend on the reasoner (see `DEFAULT_PROMPTS` in
`nl_2_fol/inference/cli.py`) and can be overwritten with `--system_prompt_fp`,
`--few_shot_prompt_fp`, `--err_failure_prompt_fp` and `--undef_failure_prompt_fp`.

The data paths can be changed with `--slim_dataset_path` and `--structures_data_path` (also for
`validate`).


## Validation

The reasoner is inferred from the learned definitions. Definitions are evaluated on all validation
samples (with ASP, the CPUs are shared between parallel validation workers).

- `--class_name "all"` validates all classes and writes
  `learned_definitions_a3_with_val_file_idx_None_.pkl` next to the input file.
- `--class_name "ethanol"` validates a single class and writes `ethanol.pkl` to the current
  working directory.
- To split the work across (HPC) jobs, pass a text file with one class name per line and a unique
  index per job; the output is `learned_definitions_a3_with_val_file_idx_<index>_.pkl`:

  ```bash
  python nl_2_fol/inference/cli.py validate \
    --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-5-5_asp/learned_definitions_a3.pkl" \
    --class_names_txt_file_path "classes_0.txt" \
    --file_save_index 0
  ```


## Utility Scripts

Scripts in `nl_2_fol/inference/utils/` for working with definition pickles:

```bash
# show all classes (or one with --class-name, plus prompts with --system-prompt / --conversation-history);
# other subcommands: delete, stats, ...
python nl_2_fol/inference/utils/show_learned_content.py --pickle-file <defs.pkl> show --class-name "ethanol"

# merge the validation metrics of <source.pkl> into <target.pkl>, write <output.pkl>
python nl_2_fol/inference/utils/merge_validation_metrics.py <target.pkl> <source.pkl> <output.pkl>

# compare with C3P scores (JSON files from https://github.com/chemkg/c3p/pull/23), export a CSV
python nl_2_fol/inference/utils/compare_with_c3p.py \
  --ensemble-c3p-json c3p_ensemble_train_val_scores.json \
  --o3-mini-c3p-json c3p_o3_mini_train_val_scores.json \
  --learned-pickle <defs_with_val.pkl> \
  --output-csv comparison_with_c3p.csv
```


## Running a Custom Model with Ollama on a Cluster

This example uses the fine-tuned Mistral FOL model
[fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol](https://huggingface.co/fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol)
with `learn_mistral` (defaults: `--api_platform ollama --model_name my-mistral`). Install
`.[ollama]`.

1. **Merge the model weights** into a local `mistral-merged` directory with `convert_mistral_to_gguf`
   from `nl_2_fol/prompting/custom_api/_to_gguf.py`.

2. **Get `llama.cpp` and Ollama** (user-local, no root needed):

   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   pip install -r llama.cpp/requirements.txt

   mkdir -p "$HOME/ollama" && cd "$HOME/ollama"
   curl -L -o ollama-linux-amd64.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
   unzstd ollama-linux-amd64.tar.zst && tar -xf ollama-linux-amd64.tar
   echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   ollama --version
   ```

3. **Convert to GGUF** (from inside `llama.cpp`; can take a while and needs a lot of RAM):

   ```bash
   python convert_hf_to_gguf.py ../mistral-merged --outfile mistral.gguf
   ```

4. **Start the Ollama server** in the background:

   ```bash
   export OLLAMA_HOST=http://localhost:<your_custom_port>
   export OLLAMA_TIMEOUT=180  # seconds
   ollama serve > ollama.log 2>&1 &
   OLLAMA_PID=$!
   ```

   Stop it afterwards with `kill $OLLAMA_PID; wait $OLLAMA_PID`.

5. **Register the model** (once per build): create a `Modelfile` containing `FROM ./mistral.gguf`
   next to `mistral.gguf`, then run

   ```bash
   ollama create my-mistral -f Modelfile
   ollama list  # shows my-mistral
   ```

6. **Learn definitions.** Proxy variables can interfere with localhost connections on some clusters:

   ```bash
   export NO_PROXY=127.0.0.1,localhost,.local
   export no_proxy=127.0.0.1,localhost,.local
   python nl_2_fol/inference/cli.py learn_mistral
   ```

   `ollama serve` and the learning command have to run on the same compute node (e.g. in the same
   job allocation).
