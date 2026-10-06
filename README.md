# GPT550MIL

A small GPT-style character-level language model trained on the Tiny Shakespeare dataset. The project is based on a compact GPT-2 implementation and is suitable for experimenting on a Mac with Apple Silicon/MPS or on CPU.

## How the files connect

```text
data/shakespeare_char/input.txt
        |
        v
 data/shakespeare_char/prepare.py
        |
        +--> train.bin
        +--> val.bin
        +--> meta.pkl
                    |
                    v
config/train_shakespeare_char.py --> train.py --> model.py
                                             |
                                             v
                              out-shakespeare-char/ckpt.pt
                                             |
                                             v
                         sample.py --> generated text
```

- `data/shakespeare_char/input.txt` is the raw training text. If it is missing, `prepare.py` downloads Tiny Shakespeare.
- `data/shakespeare_char/prepare.py` builds a character vocabulary, converts characters to integer IDs, creates a 90/10 train/validation split, and writes `train.bin`, `val.bin`, and `meta.pkl`.
- `model.py` defines the GPT configuration, transformer blocks, causal self-attention, loss calculation, optimizer setup, and text generation method.
- `train.py` loads the prepared binary data, trains a model, evaluates it, and saves checkpoints.
- `config/train_shakespeare_char.py` contains the small character-level training settings. It overrides the defaults in `train.py`.
- `configurator.py` applies a Python config file and then command-line overrides such as `--device=mps`.
- `sample.py` loads a checkpoint, encodes a prompt using `meta.pkl`, generates text, and decodes the output back to characters.
- `out-shakespeare-char/ckpt.pt` is the generated checkpoint. It is ignored by Git.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `model.py` imports Hugging Face Transformers for optional GPT-2 weight loading. Install it if it is not already available in your environment:

```bash
python -m pip install transformers
```

## Prepare the dataset

Run this once before training:

```bash
python data/shakespeare_char/prepare.py
```

This creates the following generated files inside `data/shakespeare_char/`:

- `train.bin`: encoded training split
- `val.bin`: encoded validation split
- `meta.pkl`: character-to-ID and ID-to-character mappings

These generated binary files are ignored by Git and can be recreated at any time.

## Train on Apple Silicon

The project automatically selects MPS when it is available. For a first run, disable compilation because `torch.compile` support can vary across PyTorch and MPS versions:

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

The Shakespeare configuration trains for 5,000 iterations and writes the checkpoint to `out-shakespeare-char/ckpt.pt`.

To continue training from that checkpoint:

```bash
python train.py config/train_shakespeare_char.py \
  --init_from=resume \
  --device=mps \
  --compile=False
```

## Train on CPU

Use the same configuration with CPU selected explicitly:

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

CPU training is useful for testing the pipeline but will be considerably slower.

## Generate text

After a checkpoint exists, run:

```bash
python sample.py \
  --init_from=resume \
  --out_dir=out-shakespeare-char \
  --device=mps
```

For CPU generation:

```bash
python sample.py \
  --init_from=resume \
  --out_dir=out-shakespeare-char \
  --device=cpu
```

By default, `sample.py` generates 10 samples with up to 500 new characters each. Change the prompt and sampling behavior from the command line:

```bash
python sample.py \
  --init_from=resume \
  --out_dir=out-shakespeare-char \
  --device=mps \
  --start="ROMEO:" \
  --num_samples=3 \
  --max_new_tokens=200 \
  --temperature=0.8 \
  --top_k=40
```

A prompt can also be read from a file by using `--start=FILE:path/to/prompt.txt`.

## Configuration and useful options

A config file is passed as a positional argument, followed by command-line overrides:

```bash
python train.py config/train_shakespeare_char.py --batch_size=32 --max_iters=1000
```

Common training options include:

- `--device=mps`, `--device=cpu`, or another supported PyTorch device
- `--compile=False` to disable `torch.compile`
- `--max_iters=1000` to shorten a test run
- `--eval_only=True` to evaluate once and exit
- `--out_dir=some-output-directory` to choose another checkpoint directory
- `--init_from=scratch` to start a new model or `--init_from=resume` to continue training

The command-line value must have the same basic Python type as the setting it replaces. Boolean values should be written as `True` or `False`.

## Model details

The Shakespeare configuration uses:

- Character-level tokenization with a vocabulary of about 65 characters
- A context length of 256 characters
- 6 transformer layers
- 6 attention heads
- 384-dimensional embeddings
- Dropout of 0.2
- AdamW optimization with a cosine learning-rate schedule

The model predicts the next character from the preceding context. During training, the input sequence is shifted by one character to create the target sequence. During sampling, each predicted character is appended to the context and used to predict the next one.

## Other entry points

- `model.ipynb` contains an interactive version of the model work.
- `config/` contains additional training and evaluation configurations for GPT-2-sized experiments.
- `data/` contains dataset preparation scripts and dataset-specific documentation.
- `bench.py` and `configurator.py` provide supporting utilities for experiments and configuration handling.
