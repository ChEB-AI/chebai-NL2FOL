# chebai-NL2FOL

AI workflow for natural language to First-Order Logic (FOL) translation for ChEBI.

<img width="2796" height="1024" alt="fig_landscape" src="https://github.com/user-attachments/assets/58dcf948-4645-4523-bb02-6120306063a0" />


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
