"""
plot_cross_attention.py — Research-paper quality cross-attention heatmaps.

Generates heatmaps showing how BART's decoder attends over EEG word positions
during autoregressive text generation.

    Rows    = decoded output tokens  (y-axis, top → bottom)
    Columns = input EEG word positions (x-axis, labeled with ground-truth words)

Usage:
    python plot_cross_attention.py \
        --checkpoint Results/checkpoints/best.pt \
        --data_dir dataset --n_samples 5 \
        --output_dir Results/attention_heatmaps

The script first scores ALL test samples, then picks the top-N by the chosen
metric (--rank_by: rougeL or bleu4) so the plotted heatmaps showcase the
model's best behaviour.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import (
    ZuCoEEGDataset, eeg_collate_fn,
    split_samples_by_subject, split_samples,
)
from eeg_to_text.models.eeg_to_text import EEGToTextModel


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot research-paper quality cross-attention heatmaps",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--split", type=str, default="test",
                    choices=["dev", "test"])
    p.add_argument("--split_mode", type=str, default="sentence",
                    choices=["subject", "sentence"])
    p.add_argument("--n_samples", type=int, default=5,
                    help="Number of BEST samples to plot")
    p.add_argument("--rank_by", type=str, default="rougeL",
                    choices=["rougeL", "bleu4"],
                    help="Metric to rank samples by (default: rougeL)")
    p.add_argument("--n_candidates", type=int, default=0,
                    help="How many samples to score for ranking (0 = all)")
    p.add_argument("--output_dir", type=str,
                    default="Results/attention_heatmaps")
    p.add_argument("--per_layer", action="store_true",
                    help="Also save per-layer heatmaps")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bart_model", type=str, default=None)
    p.add_argument("--task_files", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=56)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Heatmap plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_heatmap(
    attn_weights: np.ndarray,
    output_tokens: list,
    eeg_word_labels: list,
    target_text: str,
    decoded_text: str,
    scores: dict,
    save_path: str,
    title_extra: str = "",
    dpi: int = 200,
):
    """
    Research-paper quality attention heatmap.

    Args:
        attn_weights:   (T_out, L_eeg) attention weights (head- & layer-averaged).
        output_tokens:  list of decoded token strings (length T_out).
        eeg_word_labels: list of ground-truth word labels for EEG positions.
        target_text:    ground-truth sentence.
        decoded_text:   model-decoded sentence.
        scores:         dict with 'rougeL' and/or 'bleu4' float values.
        save_path:      output PNG path.
        title_extra:    optional subtitle (e.g. "Layer 3").
        dpi:            figure DPI.
    """
    n_eeg = len(eeg_word_labels)
    n_tok = len(output_tokens)
    attn = attn_weights[:n_tok, :n_eeg]

    # ── Figure sizing ───────────────────────────────────────────────────
    cell_w, cell_h = 0.55, 0.38
    fig_w = max(5.5, n_eeg * cell_w + 2.5)
    fig_h = max(4.0, n_tok * cell_h + 3.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Custom colormap (dark → light) ─────────────────────────
    cmap = plt.cm.viridis  # default matplotlib colormap, colorblind friendly

    im = ax.imshow(
        attn, aspect="auto", cmap=cmap, interpolation="nearest",
        vmin=0, vmax=attn.max() if attn.max() > 0 else 1.0,
    )

    # ── Axis labels ─────────────────────────────────────────────────────
    # Y-axis: decoded tokens (BPE pieces)
    ax.set_yticks(range(n_tok))
    cleaned_tokens = []
    for t in output_tokens:
        t = t.replace("Ġ", "").replace("▁", "").strip()
        if not t:
            t = "·"
        cleaned_tokens.append(t)
    ax.set_yticklabels(cleaned_tokens, fontsize=8.5, fontfamily="monospace")

    # X-axis: ground-truth words at EEG positions
    ax.set_xticks(range(n_eeg))
    ax.set_xticklabels(
        eeg_word_labels, fontsize=8, rotation=45, ha="right",
        fontfamily="serif",
    )
    ax.tick_params(axis="both", length=2, pad=3)

    ax.set_xlabel("Input EEG word position (ground-truth words)", fontsize=10,
                   labelpad=8)
    ax.set_ylabel("Decoded output tokens →", fontsize=10, labelpad=8)

    # ── Grid lines between cells ────────────────────────────────────────
    ax.set_xticks([x - 0.5 for x in range(1, n_eeg)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_tok)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", length=0)

    # ── Colorbar ────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, shrink=0.85)
    cbar.set_label("Attention weight", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # ── Title + annotations ─────────────────────────────────────────────
    main_title = "Cross-Attention Heatmap"
    if title_extra:
        main_title += f" — {title_extra}"
    ax.set_title(main_title, fontsize=12, fontweight="bold", pad=12)

    # Text annotations below the figure
    target_short = target_text if len(target_text) <= 90 else target_text[:87] + "…"
    decoded_short = decoded_text if len(decoded_text) <= 90 else decoded_text[:87] + "…"

    score_parts = []
    if "rougeL" in scores:
        score_parts.append(f"ROUGE-L: {scores['rougeL']:.3f}")
    if "bleu4" in scores:
        score_parts.append(f"BLEU-4: {scores['bleu4']:.4f}")
    score_line = "  |  ".join(score_parts)

    annotation = (
        f"Target:   {target_short}\n"
        f"Decoded: {decoded_short}\n"
        f"{score_line}"
    )
    fig.text(
        0.03, -0.01, annotation,
        fontsize=7.5, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.device = args.device
    cfg.seed = args.seed
    if args.bart_model:
        cfg.bart_model = args.bart_model
        if "base" in args.bart_model:
            cfg.bart_dim = 768
    if args.task_files is not None:
        cfg.task_pickle_files = args.task_files

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────
    print("Loading data...")
    dataset_dicts = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    if not dataset_dicts:
        print(f"ERROR: No datasets in {cfg.data_dir}")
        sys.exit(1)

    preprocessor = EEGPreprocessor(
        eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels,
    )
    all_samples = preprocessor.extract_all_sentences_with_subjects(
        dataset_dicts, subject=cfg.subject,
    )

    if args.split_mode == "sentence":
        plain_samples = [(e, t) for e, t, _ in all_samples]
        train_samples, dev_samples, test_samples = split_samples(
            plain_samples, seed=cfg.seed,
        )
    else:
        train_samples, dev_samples, test_samples = split_samples_by_subject(
            all_samples, seed=cfg.seed,
        )

    eval_samples = test_samples if args.split == "test" else dev_samples

    # Normalisation
    ckpt_dir = os.path.dirname(args.checkpoint)
    stats_path = os.path.join(ckpt_dir, "eeg_norm_stats.npz")
    if os.path.isfile(stats_path):
        print(f"Loading normalisation stats from {stats_path}")
        preprocessor.load_stats(stats_path)
    else:
        print("WARNING: No saved normalisation stats, fitting on train data")
        preprocessor.fit([s[0] for s in train_samples])

    eval_samples = [(preprocessor.transform(e), t) for e, t in eval_samples]

    # ── Tokenizer + Dataset ─────────────────────────────────────────────
    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model)
    eval_ds = ZuCoEEGDataset(
        eval_samples, tokenizer, cfg.max_words, cfg.max_text_len,
    )

    # ── Model ───────────────────────────────────────────────────────────
    print(f"Building model ({cfg.bart_model})...")
    model = EEGToTextModel(
        bart_model_name=cfg.bart_model,
        eeg_input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim,
        s4d_layers=cfg.s4d_layers,
        s4d_state_dim=cfg.s4d_state_dim,
        s4d_dropout=cfg.s4d_dropout,
        s4d_bidirectional=cfg.s4d_bidirectional,
        gate_bias_init=cfg.gate_bias_init,
    )
    model.to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')}  |  Metric: {ckpt.get('metric', '?')}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Score all candidates to find best samples
    # ═══════════════════════════════════════════════════════════════════
    n_candidates = args.n_candidates if args.n_candidates > 0 else len(eval_ds)
    n_candidates = min(n_candidates, len(eval_ds))

    rank_metric = args.rank_by
    print(f"\nPhase 1: Scoring {n_candidates} samples to find top-{args.n_samples} by {rank_metric}...")

    rouge_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu_sc = BLEU(max_ngram_order=4, smooth_method="exp", smooth_value=0.01)
    # Each entry: {"rougeL": float, "bleu4": float, "idx": int, "pred": str, "target": str}
    scored_samples = []

    # Score in batches
    scoring_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=0,
    )

    sample_idx = 0
    for batch in scoring_loader:
        if sample_idx >= n_candidates:
            break

        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        raw_texts = batch["raw_text"]
        B = eeg.size(0)

        # Quick greedy decode (no attention extraction needed here)
        preds = model.generate_text(
            eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
            max_length=args.max_length, num_beams=1,
            repetition_penalty=args.repetition_penalty,
        )

        for i in range(B):
            if sample_idx >= n_candidates:
                break
            # ROUGE-L
            rs = rouge_sc.score(raw_texts[i], preds[i])
            rl = rs["rougeL"].fmeasure
            # Sentence-level BLEU-4
            b4 = bleu_sc.corpus_score([preds[i]], [[raw_texts[i]]]).score / 100.0
            scored_samples.append({
                "rougeL": rl, "bleu4": b4,
                "idx": sample_idx, "pred": preds[i], "target": raw_texts[i],
            })
            sample_idx += 1

    # Sort by chosen metric descending, take top-N
    scored_samples.sort(key=lambda x: x[rank_metric], reverse=True)
    top_samples = scored_samples[:args.n_samples]

    print(f"\nTop-{args.n_samples} samples by {rank_metric}:")
    for rank, s in enumerate(top_samples):
        print(f"  #{rank+1}  ROUGE-L={s['rougeL']:.3f}  BLEU-4={s['bleu4']:.4f}  idx={s['idx']}")
        print(f"       Target:  {s['target'][:80]}{'…' if len(s['target'])>80 else ''}")
        print(f"       Decoded: {s['pred'][:80]}{'…' if len(s['pred'])>80 else ''}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Extract cross-attention for best samples & plot
    # ═══════════════════════════════════════════════════════════════════
    print(f"\nPhase 2: Extracting cross-attention and plotting heatmaps...\n")

    for rank, s in enumerate(top_samples):
        idx = s["idx"]
        target_text = s["target"]
        sample_scores = {"rougeL": s["rougeL"], "bleu4": s["bleu4"]}

        # Load the specific sample
        batch = eeg_collate_fn([eval_ds[idx]])
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        n_eeg_words = int(eeg_mask[0].sum().item())

        # Generate with attention extraction
        texts, cross_attns = model.generate_with_cross_attention(
            eeg=eeg,
            eeg_mask=eeg_mask,
            tokenizer=tokenizer,
            max_length=args.max_length,
            repetition_penalty=args.repetition_penalty,
        )
        # cross_attns: (1, n_layers, n_heads, T_out, L_eeg)

        decoded_text = texts[0]

        # Get output token labels (for y-axis of heatmap)
        output_token_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
        output_tokens = [tokenizer.decode([tid]) for tid in output_token_ids]

        # Trim to match attention steps
        T_out = cross_attns.size(3)
        if len(output_tokens) > T_out:
            output_tokens = output_tokens[:T_out]
        elif T_out > len(output_tokens) and len(output_tokens) > 0:
            cross_attns = cross_attns[:, :, :, :len(output_tokens), :]

        if len(output_tokens) == 0:
            print(f"  #{rank+1}: empty output, skipping")
            continue

        # Get EEG word position labels (from ground truth target words)
        target_words = target_text.split()
        eeg_word_labels = []
        for pos in range(n_eeg_words):
            if pos < len(target_words):
                w = target_words[pos]
                if len(w) > 12:
                    w = w[:10] + "…"
                eeg_word_labels.append(f"{pos}: {w}")
            else:
                eeg_word_labels.append(f"{pos}")

        # ── Average heatmap (all layers, all heads) ─────────────────────
        attn_avg = cross_attns[0].mean(dim=0).mean(dim=0).cpu().numpy()

        metric_val = s[rank_metric]
        fname = f"{rank_metric}_top{rank+1}_idx{idx}_{rank_metric}{metric_val:.3f}_avg.png"
        plot_heatmap(
            attn_weights=attn_avg,
            output_tokens=output_tokens,
            eeg_word_labels=eeg_word_labels,
            target_text=target_text,
            decoded_text=decoded_text,
            scores=sample_scores,
            save_path=os.path.join(args.output_dir, fname),
            dpi=args.dpi,
        )

        # ── Per-layer heatmaps ──────────────────────────────────────────
        if args.per_layer:
            n_layers = cross_attns.size(1)
            for layer_idx in range(n_layers):
                attn_layer = cross_attns[0, layer_idx].mean(dim=0).cpu().numpy()
                fname_layer = (
                    f"{rank_metric}_top{rank+1}_idx{idx}_layer{layer_idx}.png"
                )
                plot_heatmap(
                    attn_weights=attn_layer,
                    output_tokens=output_tokens,
                    eeg_word_labels=eeg_word_labels,
                    target_text=target_text,
                    decoded_text=decoded_text,
                    scores=sample_scores,
                    save_path=os.path.join(args.output_dir, fname_layer),
                    title_extra=f"Layer {layer_idx}",
                    dpi=args.dpi,
                )

    print(f"\n{'='*60}")
    print(f"Done! {args.n_samples} heatmaps saved to: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
