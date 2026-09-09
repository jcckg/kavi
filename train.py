import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.checkpoint import checkpoint

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
DEVANAGARI = [chr(i) for i in range(0x0900, 0x0980)]
SANSKRIT_PAD = 0
SANSKRIT_BOS = 1
SANSKRIT_EOS = 2
SANSKRIT_OFFSET = 3


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sinusoidal(max_len, d_model):
    encoding = torch.zeros(max_len, d_model)
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(
        positions * frequencies[: encoding[:, 1::2].shape[1]]
    )
    return encoding.unsqueeze(0)


def train_tokeniser(texts, vocab_size, path):
    tokeniser = Tokenizer(BPE(unk_token=UNK))
    tokeniser.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokeniser.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=True
    )
    tokeniser.train_from_iterator(texts, trainer=trainer)
    tokeniser.save(str(path))
    return tokeniser


def english_texts(dataset_split):
    for row in dataset_split:
        text = row["translation"]["en"].strip()
        if text:
            yield text


def split_ids(token_ids, prompt_fraction, max_prompt_tokens, max_target_tokens):
    if len(token_ids) < 2:
        return None
    prompt_len = max(1, int(len(token_ids) * prompt_fraction))
    prompt_len = min(prompt_len, max_prompt_tokens, len(token_ids) - 1)
    target = token_ids[prompt_len : prompt_len + max_target_tokens]
    if not target:
        return None
    return token_ids[:prompt_len], target


def make_examples(texts, tokeniser, args):
    examples = []
    for text in texts:
        token_ids = tokeniser.encode(text).ids
        item = split_ids(
            token_ids,
            args.prompt_fraction,
            args.max_prompt_tokens,
            args.max_target_tokens,
        )
        if item is not None:
            examples.append(item)
    return examples


def collate(batch, pad_id, bos_id, eos_id, device):
    batch_size = len(batch)
    max_prompt = max(len(x[0]) for x in batch)
    max_decoder = max(1 + len(prompt) + len(target) for prompt, target in batch)
    prompts = np.full((batch_size, max_prompt), pad_id, dtype=np.int64)
    decoder_inputs = np.full((batch_size, max_decoder), pad_id, dtype=np.int64)
    targets = np.full((batch_size, max_decoder), pad_id, dtype=np.int64)
    loss_mask = np.zeros((batch_size, max_decoder), dtype=np.float32)
    for index, (prompt, target) in enumerate(batch):
        prompt = list(prompt)
        target = list(target)
        prompts[index, : len(prompt)] = prompt
        decoder_input = [bos_id] + prompt + target
        decoder_target = prompt + target + [eos_id]
        decoder_inputs[index, : len(decoder_input)] = decoder_input
        targets[index, : len(decoder_target)] = decoder_target
        loss_mask[index, len(prompt) : len(decoder_target)] = 1.0
    return {
        "prompt": torch.tensor(prompts, device=device),
        "prompt_pad": torch.tensor(prompts == pad_id, device=device),
        "dec_input": torch.tensor(decoder_inputs, device=device),
        "dec_pad": torch.tensor(decoder_inputs == pad_id, device=device),
        "target": torch.tensor(targets, device=device),
        "loss_mask": torch.tensor(loss_mask, device=device),
    }


def causal_mask(size, device):
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


def sample_logits(logits, temperature, forbidden, top_k=0, top_p=1.0):
    values = logits.detach().float().clone() / temperature
    values[forbidden] = -float("inf")
    if top_k and top_k > 0:
        keep = min(top_k, values.numel())
        cutoff = torch.topk(values, keep).values[-1]
        values = torch.where(
            values < cutoff, torch.full_like(values, -float("inf")), values
        )
    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        sorted_probs = F.softmax(sorted_values, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values[sorted_indices[remove]] = -float("inf")
    probs = F.softmax(values, dim=-1)
    return int(torch.multinomial(probs, 1).item())


class PromptEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, heads, d_ff, layers, max_len, checkpoint_layers=False
    ):
        super().__init__()
        self.checkpoint_layers = checkpoint_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos", sinusoidal(max_len, d_model), persistent=False)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids, pad_mask):
        x = self.embed(ids) + self.pos[:, : ids.shape[1]].to(ids.device)
        for layer in self.layers:
            if self.checkpoint_layers and self.training:

                def run(y):
                    return layer(y, src_key_padding_mask=pad_mask)

                x = checkpoint(run, x, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=pad_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, heads, d_ff, layers, max_len, checkpoint_layers=False
    ):
        super().__init__()
        self.checkpoint_layers = checkpoint_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos", sinusoidal(max_len, d_model), persistent=False)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, ids=None, embeds=None, memory=None, self_pad=None, memory_pad=None
    ):
        x = self.embed(ids) if embeds is None else embeds
        x = x + self.pos[:, : x.shape[1]].to(x.device)
        mask = causal_mask(x.shape[1], x.device)
        for layer in self.layers:
            if self.checkpoint_layers and self.training:

                def run(y, mem):
                    return layer(
                        y,
                        mem,
                        tgt_mask=mask,
                        tgt_key_padding_mask=self_pad,
                        memory_key_padding_mask=memory_pad,
                    )

                x = checkpoint(run, x, memory, use_reentrant=False)
            else:
                x = layer(
                    x,
                    memory,
                    tgt_mask=mask,
                    tgt_key_padding_mask=self_pad,
                    memory_key_padding_mask=memory_pad,
                )
        x = self.norm(x)
        return x, self.head(x)


class TwoStageGenerator(nn.Module):
    def __init__(self, english_vocab, args):
        super().__init__()
        self.english_vocab = english_vocab
        self.sanskrit_vocab = len(DEVANAGARI) + SANSKRIT_OFFSET
        self.stage1_chars = args.stage1_chars
        self.train_stage1_chars = getattr(args, "train_stage1_chars", args.stage1_chars)
        checkpoint_layers = getattr(args, "checkpoint", False)
        max_english = args.max_prompt_tokens + args.max_target_tokens + 2
        self.encoder = PromptEncoder(
            english_vocab,
            args.d_model,
            args.heads,
            args.d_ff,
            2,
            args.max_prompt_tokens,
            checkpoint_layers,
        )
        self.stage1 = Decoder(
            self.sanskrit_vocab,
            args.d_model,
            args.heads,
            args.d_ff,
            3,
            args.stage1_chars + 1,
            checkpoint_layers,
        )
        self.stage2 = Decoder(
            english_vocab,
            args.d_model,
            args.heads,
            args.d_ff,
            3,
            max_english,
            checkpoint_layers,
        )

    def soft_stage1(self, prompt_memory, prompt_pad):
        batch_size = prompt_memory.shape[0]
        ids = torch.full(
            (batch_size, 1),
            SANSKRIT_BOS,
            dtype=torch.long,
            device=prompt_memory.device,
        )
        embeds = self.stage1.embed(ids)
        states = []
        for _ in range(self.train_stage1_chars):
            hidden, logits = self.stage1(
                embeds=embeds, memory=prompt_memory, memory_pad=prompt_pad
            )
            last_hidden = hidden[:, -1:, :]
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_embed = probs @ self.stage1.embed.weight
            embeds = torch.cat([embeds, next_embed.unsqueeze(1)], dim=1)
            states.append(last_hidden)
        return torch.cat(states, dim=1)

    def forward(self, batch):
        prompt_memory = self.encoder(batch["prompt"], batch["prompt_pad"])
        stage1_memory = self.soft_stage1(prompt_memory, batch["prompt_pad"])
        _, logits = self.stage2(
            ids=batch["dec_input"],
            memory=stage1_memory,
            self_pad=batch["dec_pad"],
            memory_pad=None,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target"].reshape(-1),
            reduction="none",
        ).reshape_as(batch["target"])
        loss = loss * batch["loss_mask"]
        return loss.sum() / batch["loss_mask"].sum()

    @torch.no_grad()
    def encode_prompt(self, prompt_ids, pad_id, device):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pad = torch.tensor(
            [[x == pad_id for x in prompt_ids]], dtype=torch.bool, device=device
        )
        return self.encoder(ids, pad), pad

    @torch.no_grad()
    def generate_stage1(
        self, prompt_memory, prompt_pad, temperature, max_chars=None, top_k=0, top_p=1.0
    ):
        ids = [SANSKRIT_BOS]
        chars = []
        states = []
        limit = (
            self.stage1_chars
            if max_chars is None
            else min(max_chars, self.stage1_chars)
        )
        for _ in range(limit):
            token_ids = torch.tensor(
                [ids], dtype=torch.long, device=prompt_memory.device
            )
            hidden, logits = self.stage1(
                ids=token_ids, memory=prompt_memory, memory_pad=prompt_pad
            )
            idx = sample_logits(
                logits[0, -1], temperature, [SANSKRIT_PAD, SANSKRIT_BOS], top_k, top_p
            )
            states.append(hidden[:, -1:, :])
            if idx == SANSKRIT_EOS:
                break
            ids.append(idx)
            if idx >= SANSKRIT_OFFSET:
                chars.append(DEVANAGARI[idx - SANSKRIT_OFFSET])
        if not states:
            token_ids = torch.tensor(
                [[SANSKRIT_BOS]], dtype=torch.long, device=prompt_memory.device
            )
            states.append(
                self.stage1(
                    ids=token_ids, memory=prompt_memory, memory_pad=prompt_pad
                )[0]
            )
        return "".join(chars), torch.cat(states, dim=1)

    @torch.no_grad()
    def soft_stage1_infer(self, prompt_memory, prompt_pad, max_chars):
        old_len = self.train_stage1_chars
        try:
            self.train_stage1_chars = max_chars
            return self.soft_stage1(prompt_memory, prompt_pad)
        finally:
            self.train_stage1_chars = old_len

    @torch.no_grad()
    def generate_stage2(
        self,
        prompt_ids,
        stage1_memory,
        temperature,
        max_tokens,
        min_tokens,
        pad_id,
        bos_id,
        eos_id,
        top_k=0,
        top_p=1.0,
    ):
        ids = [bos_id] + prompt_ids
        generated = []
        for _ in range(max_tokens):
            token_ids = torch.tensor(
                [ids], dtype=torch.long, device=stage1_memory.device
            )
            pad = torch.tensor(
                [[x == pad_id for x in ids]],
                dtype=torch.bool,
                device=stage1_memory.device,
            )
            _, logits = self.stage2(
                ids=token_ids, memory=stage1_memory, self_pad=pad
            )
            forbidden = [pad_id, bos_id]
            if len(generated) < min_tokens:
                forbidden.append(eos_id)
            idx = sample_logits(logits[0, -1], temperature, forbidden, top_k, top_p)
            if idx == eos_id:
                break
            ids.append(idx)
            generated.append(idx)
        return generated


def save_config(args, english_vocab, paths):
    config = {
        "english_vocab": english_vocab,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "heads": args.heads,
        "d_ff": args.d_ff,
        "stage1_chars": args.stage1_chars,
        "train_stage1_chars": getattr(args, "train_stage1_chars", args.stage1_chars),
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_target_tokens": args.max_target_tokens,
        "tokeniser": str(paths["tokeniser"]),
        "weights": str(paths["weights"]),
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")


def config_args(config):
    namespace = argparse.Namespace()
    namespace.vocab_size = config["vocab_size"]
    namespace.d_model = config["d_model"]
    namespace.heads = config["heads"]
    namespace.d_ff = config["d_ff"]
    namespace.stage1_chars = config["stage1_chars"]
    namespace.train_stage1_chars = config.get(
        "train_stage1_chars", config["stage1_chars"]
    )
    namespace.max_prompt_tokens = config["max_prompt_tokens"]
    namespace.max_target_tokens = config["max_target_tokens"]
    namespace.checkpoint = False
    return namespace


def paths(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "dir": out,
        "tokeniser": out / "itihasa_bpe_8000.json",
        "weights": out / "two_stage_generator.pt",
        "config": out / "config.json",
    }


def resolve_artifact_path(saved_path, fallback_path):
    saved = Path(saved_path)
    if saved.exists():
        return saved
    if fallback_path.exists():
        return fallback_path
    return saved


def build_scheduler(optimiser, total_steps, warmup_steps):
    def learning_rate_multiplier(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate_multiplier)


def train(args):
    artifact_paths = paths(args)
    device = pick_device(args.device)
    if args.train_stage1_chars < 1 or args.train_stage1_chars > args.stage1_chars:
        raise ValueError("--train-stage1-chars must be between 1 and --stage1-chars")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if args.checkpoint:
        print("Activation checkpointing: on")
    if args.train_stage1_chars != args.stage1_chars:
        print(
            f"Training stage 1 unroll: {args.train_stage1_chars} chars; "
            f"inference maximum: {args.stage1_chars} chars"
        )
    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.dataset)
    training_text = list(english_texts(dataset["train"]))
    if artifact_paths["tokeniser"].exists() and not args.retrain_tokeniser:
        tokeniser = Tokenizer.from_file(str(artifact_paths["tokeniser"]))
    else:
        print("Training English BPE tokeniser...")
        tokeniser = train_tokeniser(
            training_text, args.vocab_size, artifact_paths["tokeniser"]
        )
    pad_id = tokeniser.token_to_id(PAD)
    bos_id = tokeniser.token_to_id(BOS)
    eos_id = tokeniser.token_to_id(EOS)
    examples = make_examples(training_text, tokeniser, args)
    print(f"Training examples: {len(examples):,}")
    model = TwoStageGenerator(tokeniser.get_vocab_size(), args).to(device)
    if args.compile and device.type != "mps":
        model = torch.compile(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * math.ceil(
        len(examples) / (args.batch_size * args.grad_accum_steps)
    )
    scheduler = build_scheduler(optimiser, total_steps, args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(examples)
        total = 0.0
        steps = 0
        updates = 0
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        optimiser.zero_grad(set_to_none=True)
        for start in range(0, len(examples), args.batch_size):
            batch = collate(
                examples[start : start + args.batch_size],
                pad_id,
                bos_id,
                eos_id,
                device,
            )
            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=args.amp):
                    loss = model(batch)
            else:
                loss = model(batch)
            scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            total += float(loss.detach().cpu())
            steps += 1
            should_step = (
                steps % args.grad_accum_steps == 0
                or start + args.batch_size >= len(examples)
            )
            if should_step:
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                updates += 1
            if steps % args.log_every == 0:
                elapsed = max(time.time() - epoch_start, 1e-9)
                ex_per_sec = min(steps * args.batch_size, len(examples)) / elapsed
                memory_status = ""
                if device.type == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                    memory_status = f" peak_cuda {peak_memory:.2f}GB"
                mean_loss = total / steps
                learning_rate = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {steps} update {updates} "
                    f"loss {mean_loss:.4f} lr {learning_rate:.2e} "
                    f"{ex_per_sec:.1f} ex/s{memory_status}"
                )
        elapsed = max(time.time() - epoch_start, 1e-9)
        mean_loss = total / max(steps, 1)
        print(
            f"epoch {epoch} loss {mean_loss:.4f} updates {updates} "
            f"time {elapsed / 60:.1f} min"
        )
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(raw_model.state_dict(), artifact_paths["weights"])
        save_config(args, tokeniser.get_vocab_size(), artifact_paths)
    print(f"Saved weights to {artifact_paths['weights']}")


def infer(args):
    artifact_paths = paths(args)
    device = pick_device(args.device)
    config = json.loads(artifact_paths["config"].read_text(encoding="utf-8"))
    tokeniser_path = resolve_artifact_path(
        config["tokeniser"], artifact_paths["tokeniser"]
    )
    weights_path = resolve_artifact_path(config["weights"], artifact_paths["weights"])
    tokeniser = Tokenizer.from_file(str(tokeniser_path))
    model_args = config_args(config)
    model = TwoStageGenerator(config["english_vocab"], model_args).to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    prompt = sys.stdin.read().strip()
    prompt_ids = tokeniser.encode(prompt).ids[-model_args.max_prompt_tokens :]
    if not prompt_ids:
        prompt_ids = [tokeniser.token_to_id(BOS)]
    prompt_memory, prompt_pad = model.encode_prompt(
        prompt_ids, tokeniser.token_to_id(PAD), device
    )
    stage1_infer_chars = args.stage1_infer_chars
    if stage1_infer_chars is None:
        stage1_infer_chars = min(model_args.stage1_chars, model_args.train_stage1_chars)
    sanskrit, hard_stage1_memory = model.generate_stage1(
        prompt_memory,
        prompt_pad,
        args.stage1_temperature,
        stage1_infer_chars,
        args.stage1_top_k,
        args.stage1_top_p,
    )
    if args.soft_stage1_infer:
        stage1_memory = model.soft_stage1_infer(
            prompt_memory, prompt_pad, stage1_infer_chars
        )
    else:
        stage1_memory = hard_stage1_memory
    english_ids = model.generate_stage2(
        prompt_ids,
        stage1_memory,
        args.stage2_temperature,
        args.infer_tokens,
        args.min_infer_tokens,
        tokeniser.token_to_id(PAD),
        tokeniser.token_to_id(BOS),
        tokeniser.token_to_id(EOS),
        args.stage2_top_k,
        args.stage2_top_p,
    )
    prose = tokeniser.decode(english_ids)
    print(sanskrit)
    print()
    print(prose.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rahular/itihasa")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--retrain-tokeniser", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--stage1-chars", type=int, default=160)
    parser.add_argument("--train-stage1-chars", type=int, default=160)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-target-tokens", type=int, default=200)
    parser.add_argument("--prompt-fraction", type=float, default=0.25)
    parser.add_argument("--stage1-temperature", type=float, default=1.1)
    parser.add_argument("--stage2-temperature", type=float, default=0.9)
    parser.add_argument("--stage1-top-k", type=int, default=0)
    parser.add_argument("--stage2-top-k", type=int, default=0)
    parser.add_argument("--stage1-top-p", type=float, default=1.0)
    parser.add_argument("--stage2-top-p", type=float, default=1.0)
    parser.add_argument("--stage1-infer-chars", type=int)
    parser.add_argument(
        "--hard-stage1-infer", dest="soft_stage1_infer", action="store_false"
    )
    parser.set_defaults(soft_stage1_infer=True)
    parser.add_argument("--infer-tokens", type=int, default=200)
    parser.add_argument("--min-infer-tokens", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.random_seed:
        args.seed = random.SystemRandom().randrange(2**31)
        print(f"seed {args.seed}", file=sys.stderr)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    if args.infer_only:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
