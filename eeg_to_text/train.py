"""
train.py — Entry point for training the EEG-to-Text model.

Usage:
    python -m eeg_to_text.train --data_dir dataset --checkpoint_dir runs/ckpts
    python -m eeg_to_text.train --resume runs/ckpts/last.pt

Run from the workspace root (c:/kaggle2/EEG-To-text/).
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples_by_subject, split_samples
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.training.trainer import Trainer
from eeg_to_text.evaluation.metrics import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG-to-Text model")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing ZuCo pickle files")
    parser.add_argument("--checkpoint_dir", type=str, default="eeg_to_text_runs/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="eeg_to_text_runs/logs",
                        help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--phase1_epochs", type=int, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)
    parser.add_argument("--phase1_lr", type=float, default=None)
    parser.add_argument("--phase2_lr", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use mixed precision training")
    parser.add_argument("--no_fp16", action="store_true", default=False,
                        help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bart_model", type=str, default=None,
                        help="BART model name (e.g. facebook/bart-base for smaller runs)")
    parser.add_argument("--phase2_only", action="store_true", default=False,
                        help="Skip Phase 1, load checkpoint and run Phase 2 only")
    parser.add_argument("--split_mode", type=str, default="subject",
                        choices=["subject", "sentence"],
                        help="Data split strategy: 'subject' (same sents, diff subjects) "
                             "or 'sentence' (no sentence overlap, critique-paper compliant)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build config ────────────────────────────────────────────────────
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.log_dir = args.log_dir
    cfg.seed = args.seed
    cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    if args.phase1_lr is not None:
        cfg.phase1_lr = args.phase1_lr
    if args.phase2_lr is not None:
        cfg.phase2_lr = args.phase2_lr
    if args.bart_model is not None:
        cfg.bart_model = args.bart_model
        # Adjust bart_dim for bart-base
        if "base" in args.bart_model:
            cfg.bart_dim = 768
    if args.no_fp16:
        cfg.fp16 = False
    elif args.fp16:
        cfg.fp16 = True

    set_seed(cfg.seed)

    # ── Load data ───────────────────────────────────────────────────────
    print("Loading pickle datasets...")
    dataset_dicts = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    if not dataset_dicts:
        print(f"ERROR: No datasets found in {cfg.data_dir}. "
              f"Expected files: {cfg.task_pickle_files}")
        sys.exit(1)

    # ── Preprocess ──────────────────────────────────────────────────────
    print("Extracting EEG features...")
    preprocessor = EEGPreprocessor(
        eeg_type=cfg.eeg_type,
        bands=cfg.bands,
        n_channels=cfg.n_channels,
    )
    all_samples = preprocessor.extract_all_sentences_with_subjects(dataset_dicts, subject=cfg.subject)

    if len(all_samples) == 0:
        print("ERROR: No valid samples extracted. Check pickle files.")
        sys.exit(1)

    # Split data
    if args.split_mode == "sentence":
        # Sentence-level split: NO sentence overlap between train/dev/test
        # Required by critique papers to prevent BART memorising test sentences
        plain_samples = [(e, t) for e, t, s in all_samples]
        train_samples, dev_samples, test_samples = split_samples(
            plain_samples, train_ratio=0.8, dev_ratio=0.1, seed=cfg.seed)
        print(f"[INFO] Using SENTENCE-LEVEL split (no overlap) - critique-compliant")
    else:
        # Subject-level split: same sentences across different subjects
        train_samples, dev_samples, test_samples = split_samples_by_subject(
            all_samples, seed=cfg.seed)
        print(f"[INFO] Using SUBJECT-LEVEL split")

    # Fit normalisation on training data only
    train_eegs = [s[0] for s in train_samples]
    preprocessor.fit(train_eegs)

    # Transform all splits
    train_samples = [(preprocessor.transform(e), t) for e, t in train_samples]
    dev_samples = [(preprocessor.transform(e), t) for e, t in dev_samples]
    test_samples = [(preprocessor.transform(e), t) for e, t in test_samples]

    # Save normalisation stats
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    preprocessor.save_stats(os.path.join(cfg.checkpoint_dir, "eeg_norm_stats.npz"))

    # ── Tokenizer ───────────────────────────────────────────────────────
    print(f"Loading tokenizer: {cfg.bart_model}")
    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model)

    # ── Datasets & dataloaders ──────────────────────────────────────────
    train_ds = ZuCoEEGDataset(
        train_samples, tokenizer, cfg.max_words, cfg.max_text_len,
        augment=True,
        noise_std=cfg.eeg_noise_std,
        channel_drop=cfg.eeg_channel_drop,
        time_shift=cfg.eeg_time_shift,
    )
    dev_ds = ZuCoEEGDataset(dev_samples, tokenizer, cfg.max_words, cfg.max_text_len)
    test_ds = ZuCoEEGDataset(test_samples, tokenizer, cfg.max_words, cfg.max_text_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    # ── Model ───────────────────────────────────────────────────────────
    print(f"Building model with BART backbone: {cfg.bart_model}")
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Evaluation callback ─────────────────────────────────────────────
    def eval_fn(model, loader, tok, dev):
        return evaluate_model(
            model, loader, tok, dev,
            num_beams=cfg.num_beams,
            max_gen_length=cfg.max_gen_length,
            print_examples=3,
            eeg_prior_alpha=cfg.gen_eeg_prior_alpha,
            repetition_penalty=cfg.repetition_penalty,
            gen_do_sample=False,          # Skip nucleus during training (save time)
            gen_top_p=cfg.gen_top_p,
            gen_temperature=cfg.gen_temperature,
            best_of_n=0,                  # Skip BoN during training (too slow)
        )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        tokenizer=tokenizer,
        config=cfg,
        evaluate_fn=eval_fn,
    )

    # ── Train ───────────────────────────────────────────────────────────
    if args.phase2_only:
        # Load checkpoint and run Phase 2 only (skip Phase 1 warm-up)
        ckpt_path = args.resume or os.path.join(cfg.checkpoint_dir, "last.pt")
        trainer.train(resume_path=ckpt_path, phase2_only=True)
    else:
        trainer.train(resume_path=args.resume)

    # ── Final test evaluation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    if os.path.isfile(best_path):
        trainer.load_checkpoint(best_path)
    else:
        print("No best checkpoint found, using last model state")

    # Ensure self-attention dampening is active for evaluation
    model.self_attn_scale = cfg.self_attn_scale

    test_metrics = evaluate_model(
        model, test_loader, tokenizer,
        torch.device(cfg.device if torch.cuda.is_available() else "cpu"),
        num_beams=cfg.num_beams,
        max_gen_length=cfg.max_gen_length,
        print_examples=8,
        eeg_prior_alpha=cfg.gen_eeg_prior_alpha,
        repetition_penalty=cfg.repetition_penalty,
        gen_do_sample=cfg.gen_do_sample,
        gen_top_p=cfg.gen_top_p,
        gen_temperature=cfg.gen_temperature,
        best_of_n=cfg.best_of_n,
        best_of_n_temperature=cfg.best_of_n_temperature,
    )

    # Save test results
    results_path = os.path.join(cfg.checkpoint_dir, "test_results.txt")
    with open(results_path, "w") as f:
        for k, v in sorted(test_metrics.items()):
            f.write(f"{k}: {v}\n")
    print(f"\nTest results saved to {results_path}")


if __name__ == "__main__":
    main()
