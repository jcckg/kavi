# kavi

Kavi is a two-stage autoregressive prose generator implemented in PyTorch. 
It encodes an English prompt, generates an intermediate character-level 
latent sequence constrained to the Devanagari alphabet, and conditions an 
English BPE decoder on the resulting representation to generate prose. 

The model is trained on the English Dutt translations from `rahular/itihasa`, 
using a learned BPE tokeniser, a prompt encoder, and two autoregressive decoders. 
CUDA is used when available, followed by MPS and CPU.

### Setup -> Inference

```sh
uv sync
uv run kavi train --batch-size 128 --train-stage1-chars 32 # (or train w/ the colab notebook)
uv run kavi infer "The king entered the forest" # (add --random-seed for random samples per run)
```

For longer generation:

```sh
uv run kavi infer "The king entered the forest" --stage1-infer-chars 32 --infer-tokens 300 --min-infer-tokens 60
```

The included `kavi_colab.ipynb` runs the faster CUDA training path with
`--batch-size 128 --grad-accum-steps 1 --train-stage1-chars 32`, and saves
artefacts to Google Drive.

### Video Rendering

Kavi’s video renderer visualises the model’s generative trajectory by projecting 
hidden states into three-dimensional space and deforming the resulting geometry according 
to activation, attention, disagreement, and token entropy.

https://github.com/user-attachments/assets/95433869-6389-4200-88e3-5f9f0839a2bc

Common output controls:

``` sh
uv run kavi render "The king entered the forest" \
    --video-out kavi.mp4 \
    --width 1920 \
    --height 1080 \
    --fps 24 \
    --crf 15
```

Geometry and motion can be adjusted with:

- --extent — overall scale of the hidden-state object.
- --base-z-scale — depth assigned to the third principal component.
- --knn-k — semantic neighbours connected per hidden-state vertex.
- --activation-expand and --activation-lift — deformation driven by current activation.
- --fracture-gain — separation introduced by attention disagreement and entropy.
- --fov, --camera-distance, and --elevation — camera composition.
- --azimuth-drift — gradual camera rotation between model states.
- --temporal-inertia — persistence of geometry across frames.
