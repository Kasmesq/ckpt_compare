import os, argparse
import torch, torch.nn.functional as F, torch.optim as optim
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel

# --- optional: your DataStates bridge ---
try:
    from datastates_bridge import DeepSpeedDataStatesAdapter
except Exception:
    DeepSpeedDataStatesAdapter = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p = deepspeed.add_config_arguments(p)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--ckpt_dir", type=str, default="ckpts")
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--load_tag", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # 1) Tiny GPT-2 config (fits easily)
    cfg = GPT2Config(
        vocab_size=4096,
        n_positions=256,
        n_ctx=256,
        n_layer=4,
        n_head=8,
        n_embd=512,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg).to(device)

    # 2) Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3) DeepSpeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
    )

    # 4) Optional: hook your DataStates bridge (safe no-op if not available)
    if DeepSpeedDataStatesAdapter is not None:
        bridge = DeepSpeedDataStatesAdapter()

        def _try_save(save_dir, tag=None, **kw):
            try:
                return bridge.save(model_engine, save_dir, tag, **kw)
            except AttributeError:
                if model_engine.global_rank == 0:
                    print("[DataStatesBridge] save() not implemented on handle; skipping", flush=True)
                return None

        def _try_load(load_dir, tag=None, **kw):
            try:
                return bridge.load(model_engine, load_dir, tag, **kw)
            except AttributeError:
                if model_engine.global_rank == 0:
                    print("[DataStatesBridge] load() not implemented on handle; skipping", flush=True)
                return None

        model_engine.save_checkpoint = _try_save
        model_engine.load_checkpoint = _try_load

    # 5) (Optional) resume
    if args.load_dir and args.load_tag and os.path.isdir(args.load_dir):
        try:
            model_engine.load_checkpoint(args.load_dir, tag=args.load_tag)
            if model_engine.global_rank == 0:
                print(f"Resumed from {args.load_dir} tag={args.load_tag}", flush=True)
        except Exception as e:
            if model_engine.global_rank == 0:
                print(f"WARNING: resume failed ({type(e).__name__}): {e}", flush=True)

    # 6) Tiny random-LM loop
    B, T, V = 4, 128, cfg.vocab_size
    for step in range(1, args.steps + 1):
        tokens  = torch.randint(3, V, (B, T), device=device)
        inputs  = torch.cat([torch.full((B,1), cfg.eos_token_id, device=device), tokens[:,:-1]], dim=1)
        targets = tokens

        out = model_engine(input_ids=inputs)
        logits = out.logits  # [B,T,V]

        loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=cfg.pad_token_id
        )

        model_engine.backward(loss)
        model_engine.step()

        if model_engine.global_rank == 0 and step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f}", flush=True)

        if step % args.save_every == 0:
            tag = f"global_step{step}"
            try:
                model_engine.save_checkpoint(args.ckpt_dir, tag=tag)
                if model_engine.global_rank == 0:
                    print(f"saved checkpoint: {args.ckpt_dir} / {tag}", flush=True)
            except Exception as e:
                if model_engine.global_rank == 0:
                    print(f"WARNING: save skipped ({type(e).__name__}): {e}", flush=True)

if __name__ == "__main__":
    main()