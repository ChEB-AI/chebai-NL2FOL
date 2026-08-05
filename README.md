# chebai-NL2FOL

AI workflow for natural language to First-Order Logic (FOL) translation for ChEBI.

**Abstract:**

Chemical ontologies such as the Chemical Entities of Biological Interest (ChEBI) provide
structured taxonomies for organizing chemical knowledge, but manually extending these
taxonomies to new molecules remains a scalability challenge. We propose ChEBI2FOL,
a pipeline for automatically translating natural language definitions of chemical classes in
ChEBI into First-Order Logic (FOL) for rule-based and explainable chemical classification. ChEBI2FOL uses multi-shot LLM prompts with multiple feedback loops to generate
and automatically validate FOL definitions. The result is a classification system that is by
design modular (auxiliary predicates are automatically generated and shared across defi-
nitions) and platform-independent (the definitions come in a standard FOL syntax which
can be parsed by various tools). Besides classification, the definitions can also be used to
verify ontological relations directly and identify new relations.
We evaluate ChEBI2FOL on 367 ChEBI classes using three LLMs for FOL generation.
Among these models, Claude Opus 4.6 achieves the strongest performance. In comparison
to C3PO, a suite of LLM-generated classifiers in Python, ChEBI2FOL achieves a higher
recall, but lower precision, suggesting that the learned definitions are more general.
Our analysis identifies challenges for FOL-based chemical classification. Overall, these
results demonstrate the feasibility of natural language to logic translation for chemical
classification.

<img width="2796" height="1024" alt="fig_landscape" src="https://github.com/user-attachments/assets/58dcf948-4645-4523-bb02-6120306063a0" />


## Data Files

The learning and validation pipelines expect the C3PO slim dataset files under `data/` by default:

```text
data/classes_slim.csv
data/structures.csv
dataset.json
```

Download them from the C3PO dataset on Hugging Face: https://huggingface.co/datasets/MonarchInit/C3PO/tree/main


These are the same source links referenced in `nl_2_fol/inference/cli.py` and `nl_2_fol/inference/preprocessing/c3po_slim_data.py`. The C3PO dataset is associated with https://github.com/chemkg/c3p.

If your files live somewhere else, pass explicit paths to the learning or validation commands:

```bash
python nl_2_fol/inference/cli.py learn \
  --slim_dataset_path "/path/to/classes_slim.csv" \
  --structures_data_path "/path/to/structures.csv"
```

```bash
python nl_2_fol/inference/cli.py validate \
  --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  --class_name "all" \
  --slim_dataset_path "/path/to/classes_slim.csv" \
  --structures_data_path "/path/to/structures.csv"
```

The C3P comparison utilities also expect score JSON files from the C3P train/validation score output referenced in the utility help text: https://github.com/chemkg/c3p/pull/23


## Start the Learning Pipeline

Run commands from the repository root so the default `data/` and prompt-template paths resolve correctly.

To learn definitions with the default Anthropic configuration:

```bash
python nl_2_fol/inference/cli.py learn
```

To learn definitions with the local Ollama Mistral configuration:

```bash
python nl_2_fol/inference/cli.py learn_mistral
```

To learn a single ChEBI class instead of all classes:

```bash
python nl_2_fol/inference/cli.py learn --class_name "ethanol"
python nl_2_fol/inference/cli.py learn_mistral --class_name "ethanol"
```

Useful options:

```bash
python nl_2_fol/inference/cli.py learn \
  --api_platform "anthropic" \
  --model_name "claude-opus-4-6" \
  --max_attempts 3 \
  --f1_threshold 0.8
```

Learning output is saved under:

```text
nl_2_fol/inference/learner/learned/<model_name>/learned_definitions_a<max_attempts>.pkl
```

For example, with `model_name="claude-opus-4-6"` and `max_attempts=3`, the definitions file is:

```text
nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl
```

## Start the Validation Pipeline

After learning has produced a definitions pickle, validate the learned definitions with:

```bash
python nl_2_fol/inference/cli.py validate \
  --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  --class_name "all"
```

To validate only one class:

```bash
python nl_2_fol/inference/cli.py validate \
  --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  --class_name "ethanol"
```

Single-class validation writes a small result pickle named after the resolved class in the current working directory, for example `ethanol.pkl`.

For HPC or long validation runs, split the work across jobs by passing a text file with one class name per line. Use a unique `file_save_index` for each job:

```bash
python nl_2_fol/inference/cli.py validate \
  --defs_file_path "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  --class_names_txt_file_path "classes_0.txt" \
  --file_save_index 0
```

Full or split validation writes a new definitions pickle next to the input file, for example:

```text
nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val_file_idx_None_.pkl
```

When `class_names_txt_file_path` is used, the index appears in the file name, for example:

```text
nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val_file_idx_0_.pkl
```

Use `--help` to inspect the full set of options:

```bash
python nl_2_fol/inference/cli.py learn --help
python nl_2_fol/inference/cli.py learn_mistral --help
python nl_2_fol/inference/cli.py validate --help
```

## Utility Scripts

Helper scripts for inspecting, editing, merging, and comparing learned definitions live in:

```text
nl_2_fol/inference/utils/
```

Most scripts expect paths to learned definition pickles produced by the learning or validation pipeline.

### Inspect or Edit Learned Definitions

Use `show_learned_content.py` to inspect a learned definitions pickle:

```bash
python nl_2_fol/inference/utils/show_learned_content.py \
  --pickle-file "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  show
```

Show one class:

```bash
python nl_2_fol/inference/utils/show_learned_content.py \
  --pickle-file "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  show \
  --class-name "ethanol"
```

Include prompt history while inspecting a class:

```bash
python nl_2_fol/inference/utils/show_learned_content.py \
  --pickle-file "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  show \
  --class-name "ethanol" \
  --system-prompt \
  --conversation-history
```

### Merge Validation Metrics

Use `merge_validation_metrics.py` to merge validation metrics from one validated pickle into another definitions pickle:

```bash
python nl_2_fol/inference/utils/merge_validation_metrics.py \
  "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3.pkl" \
  "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val_file_idx_0_.pkl" \
  "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_merged.pkl"
```

The first path is the target/base pickle, the second path is the source pickle containing validation metrics, and the third path is the output pickle.

### Compare With C3P

Use `compare_with_c3p.py` to compare validated learned definitions against C3P score JSON files and export a CSV:

```bash
python nl_2_fol/inference/utils/compare_with_c3p.py \
  --ensemble-c3p-json "c3p_ensemble_train_val_scores.json" \
  --o3-mini-c3p-json "c3p_o3_mini_train_val_scores.json" \
  --learned-pickle "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val_file_idx_0_.pkl" \
  --output-csv "comparison_with_c3p_ensemble_o3_mini.csv"
```



## Guide: Run a custom model with Ollama on a computing cluster

This example uses the Mistral FOL model:
https://huggingface.co/fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol

## 1. Prepare model weights for conversion

Convert the Mistral model to a merged format by calling `convert_mistral_to_gguf` from:
`nl_2_fol/prompting/custom_api/_to_gguf.py`

Why this step matters:
- Hugging Face checkpoints are often split across multiple files.
- The conversion pipeline expects a clean merged model directory as input.

What this step does:
- Collects and organizes model artifacts into a local `mistral-merged` folder.
- Ensures the tokenizer/config/weights are in a format that `llama.cpp` conversion can read.

Expected result:
- A `mistral-merged` directory exists in your workspace and is ready for GGUF conversion.


## 2. Build tools and install local Ollama (no root required)

This step prepares two required components:
- `llama.cpp`, which provides the `convert_hf_to_gguf.py` conversion script.
- A user-local Ollama installation, useful on clusters where you do not have `sudo` access.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

If you do not have root access on the HPC cluster, install Ollama in your home directory:

```bash
mkdir -p "$HOME/ollama"
cd "$HOME/ollama"
curl -L -o ollama-linux-amd64.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
unzstd ollama-linux-amd64.tar.zst
tar -xf ollama-linux-amd64.tar

echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
ollama --version
```

Expected result:
- `ollama --version` prints a version string.
- You can run `ollama` commands without system-wide installation.

## 3. Convert model to GGUF

From inside `llama.cpp`:

```bash
python convert_hf_to_gguf.py ../mistral-merged --outfile mistral.gguf
```

Why this step matters:
- Ollama loads local models through GGUF files.
- This command translates the merged Hugging Face model into a runtime format Ollama can serve.

Expected result:
- A file named `mistral.gguf` is created.
- The conversion may take time and use significant CPU/RAM depending on model size.

## 4. Start Ollama server

Run the Ollama server in background using below command, so it keeps running while you execute your script or commands in same terminal.



```bash
export OLLAMA_HOST=http://localhost:<your_custom_port>
export OLLAMA_TIMEOUT=180 # in seconds
ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!
```

After you are done with ollama, cleanly stop ollama server using below commands

```bash
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null
```

## 5. Register the model in Ollama

Create a `Modelfile` in the directory containing `mistral.gguf` with:

```text
FROM ./mistral.gguf
```

Then run:

```bash
ollama create my-mistral -f Modelfile
ollama list
```

Why this step matters:
- `ollama create` registers your GGUF file under a model name (`my-mistral`).
- After registration, you can refer to the model by name in CLI calls.

Expected result:
- `ollama list` shows `my-mistral`.
- You only need to run `ollama create ...` once per model build.

## 6. Run NL-to-FOL inference with Ollama

This final step sends requests from your project CLI to the locally running Ollama server.
On some clusters, proxy variables can interfere with localhost routing, so unset them first if needed.

```bash
export NO_PROXY=127.0.0.1,localhost,.local
export no_proxy=127.0.0.1,localhost,.local
```

Then run:

```bash
python nl_2_fol/inference/cli.py --api_platform="ollama" --model_name="my-mistral"
```

**IMPORTANT:** Ensure `ollama serve` and the inference command run on the same compute node or same allocated job/session if applicable.
For example, if `ollama serve` started on `hpc3-52` but the inference command runs on `hpc3-54`, the connection might fail.

Expected result:
- The CLI connects to your local Ollama instance.
- The `my-mistral` model is used for NL-to-FOL inference.
