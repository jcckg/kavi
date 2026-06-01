# kavi

Two-stage autoregressive prose generator in PyTorch.

```sh
uv run --with torch --with 'datasets<4' --with tokenizers python train.py
```

The script loads `rahular/itihasa` from Hugging Face, uses Devanagari character
ids for Sanskrit, trains an 8,000 item BPE tokeniser on the English Dutt
translations, and trains a small prompt encoder plus two autoregressive
decoders for 4 epochs. CUDA is used when available, then MPS, then CPU.

Sample after training:

```sh
printf "The king entered the forest" | uv run --with torch --with 'datasets<4' --with tokenizers python train.py --infer-only
```

Longer inference:

```sh
printf "The king entered the forest" | uv run --with torch --with 'datasets<4' --with tokenizers python train.py --infer-only --stage1-infer-chars 32 --infer-tokens 300 --min-infer-tokens 60
```

More conservative sampling:

```sh
printf "Krishna" | uv run --with torch --with 'datasets<4' --with tokenizers python train.py --infer-only --out-dir artifacts --stage1-infer-chars 32 --stage2-temperature 0.65 --stage2-top-k 40 --infer-tokens 220
```

Different random sample:

```sh
printf "Krishna" | uv run --with torch --with 'datasets<4' --with tokenizers python train.py --infer-only --out-dir artifacts --random-seed
```

For Colab, choose a GPU runtime and run the same command. Add `--compile` on CUDA
if the session has enough time for the first compilation pass.

The included `kavi_colab.ipynb` runs the faster CUDA training path with
`--batch-size 128 --grad-accum-steps 1 --train-stage1-chars 32`, and saves
artefacts to Google Drive.
