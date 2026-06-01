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

For Colab, choose a GPU runtime and run the same command. Add `--compile` on CUDA
if the session has enough time for the first compilation pass.

The included `kavi_colab.ipynb` runs the faster CUDA training path with
`--batch-size 128 --grad-accum-steps 1 --train-stage1-chars 32`, and saves
artefacts to Google Drive.
