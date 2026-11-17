"""One GPU run: preflight, fresh fine-tuning, dev selection, final test."""
import gc
import json
import math
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from load_data import load_t5_data, normal_collate_fn
from t5_utils import (
    initialize_model, initialize_optimizer, initialize_optimizer_and_scheduler,
    load_model_from_checkpoint, save_model,
)
from train_t5 import train, train_epoch, eval_epoch, test_inference
from utils import set_random_seeds


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def preflight(args, train_loader):
    """Train one batch of the longest targets, check gradients and save/reload."""
    print("STAGE 1: GPU preflight on long training examples", flush=True)
    dataset = train_loader.dataset
    indices = sorted(range(len(dataset)),
                     key=lambda i: len(dataset.decoder_ids[i]), reverse=True)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=indices[:args.batch_size], collate_fn=normal_collate_fn)
    model = initialize_model(args)
    optimizer = initialize_optimizer(args, model)
    before = model.shared.weight.detach().clone()
    loss = train_epoch(args, model, loader, optimizer, None)
    assert math.isfinite(loss), "Nonfinite training loss"
    assert not torch.equal(before, model.shared.weight), "Parameters did not update"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del before
    model.eval()
    ids, mask, _, _, bos = next(iter(loader))
    with torch.no_grad():
        generated = model.generate(
            input_ids=ids[:2].cuda(), attention_mask=mask[:2].cuda(),
            decoder_input_ids=bos[:2].cuda(), decoder_start_token_id=dataset.bos_id,
            max_new_tokens=4, num_beams=1, do_sample=False,
        )
    assert (generated[:, 0] == dataset.bos_id).all(), "Wrong decoder start token"
    with tempfile.TemporaryDirectory(prefix="preflight-", dir="checkpoints") as folder:
        save_model(folder, model, best=True)
        restored = type(model).from_pretrained(Path(folder) / "best")
        assert all(torch.equal(p.detach().cpu(), restored.state_dict()[name])
                   for name, p in model.state_dict().items()), "Checkpoint mismatch"
        del restored
    write_json("preflight.json", {
        "passed": True, "loss": loss,
        "max_sql_tokens_including_eos": max(len(x) - 1 for x in dataset.decoder_ids),
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
    })
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Preflight passed; discard its weights and reset seed before full training.", flush=True)


def evaluate(args, model, loader, suffix):
    values = eval_epoch(
        args, model, loader, "data/dev.sql", f"results/{suffix}.sql",
        "records/ground_truth_dev.pkl", f"records/{suffix}.pkl",
    )
    result = dict(zip(["loss", "record_f1", "record_em", "sql_em", "error_rate"],
                      map(float, values)))
    assert all(math.isfinite(v) for v in result.values()), "Nonfinite dev metric"
    return result


def verify_outputs(split, suffix, expected):
    queries = Path(f"results/{suffix}.sql").read_text().splitlines()
    with open(f"records/{suffix}.pkl", "rb") as handle:
        records, errors = pickle.load(handle)
    assert len(queries) == len(records) == len(errors) == expected, f"{split} count mismatch"
    return queries, records, errors


def main():
    os.chdir(Path(__file__).resolve().parent)
    config = json.loads(Path("train_config.json").read_text())
    args = SimpleNamespace(**config)
    assert torch.cuda.is_available(), "Run this entry point on the allocated GPU node."
    assert args.finetune and args.max_n_epochs > 0 and args.patience_epochs > 0
    checkpoint = Path(f"checkpoints/ft_experiments/{args.experiment_name}")
    if checkpoint.exists():
        raise RuntimeError("Existing experiment: use a new run folder; do not overwrite it.")
    for folder in ["results", "records", "checkpoints"]:
        Path(folder).mkdir(exist_ok=True)
    write_json("run_config.json", config)
    set_random_seeds(args.seed)
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    preflight(args, train_loader)

    print("STAGE 2: Fresh pretrained T5-small; full fine-tuning", flush=True)
    set_random_seeds(args.seed)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    train(args, model, train_loader, dev_loader, optimizer, scheduler)
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    print("STAGE 3: Compare decoding on dev only", flush=True)
    model = load_model_from_checkpoint(args, best=True)
    comparison = []
    for beams in args.dev_beam_candidates:
        args.num_beams = beams
        suffix = f"t5_ft_experiment_dev_beam{beams}"
        metrics = evaluate(args, model, dev_loader, suffix)
        verify_outputs("dev", suffix, len(dev_loader.dataset))
        comparison.append({"num_beams": beams, **metrics})
        write_json("dev_comparison.json", comparison)
        print(comparison[-1], flush=True)
    # Stable max chooses greedy on an exact F1 tie.
    selected = max(comparison, key=lambda item: item["record_f1"])
    args.num_beams = selected["num_beams"]
    for folder, extension in [("results", "sql"), ("records", "pkl")]:
        shutil.copyfile(f"{folder}/t5_ft_experiment_dev_beam{args.num_beams}.{extension}",
                        f"{folder}/t5_ft_experiment_dev.{extension}")
    write_json("selected_config.json", {**config, "num_beams": args.num_beams,
                                       "dev_metrics": selected})
    model.generation_config.decoder_start_token_id = dev_loader.dataset.bos_id
    model.generation_config.num_beams = args.num_beams
    model.generation_config.max_new_tokens = 1024
    model.generation_config.do_sample = False
    model.generation_config.save_pretrained(checkpoint / "best")
    dev_loader.dataset.tokenizer.save_pretrained(checkpoint / "best")

    print("STAGE 4: Generate test outputs once with the selected dev configuration", flush=True)
    test_inference(args, model, test_loader, "results/t5_ft_experiment_test.sql",
                   "records/t5_ft_experiment_test.pkl")
    verify_outputs("test", "t5_ft_experiment_test", len(test_loader.dataset))
    # Preserve per-example dev evidence for the Q6 error analysis.
    queries, _, errors = verify_outputs("dev", "t5_ft_experiment_dev", len(dev_loader.dataset))
    questions = Path("data/dev.nl").read_text().splitlines()
    gold = Path("data/dev.sql").read_text().splitlines()
    write_json("dev_examples.json", [
        {"index": i, "question": q, "gold_sql": g, "predicted_sql": p, "execution_error": e}
        for i, (q, g, p, e) in enumerate(zip(questions, gold, queries, errors))
    ])
    write_json("completion.json", {"complete": True, "test_examples": len(test_loader.dataset),
                                   "selected_dev_metrics": selected,
                                   "checkpoint": str(checkpoint / "best")})
    print("ALL STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
