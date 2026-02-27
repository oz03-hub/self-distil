import os
import argparse
import torch
from torch.utils.data import DataLoader
import wandb

from util import TRECDataset
from bi_encoder import BiEncoder

from dotenv import load_dotenv
load_dotenv()


def train(
    dataloader: DataLoader, model: BiEncoder, optimizer, scheduler, args, device="cpu"
):
    model.train()

    print(len(dataloader))

    step = 0
    for epoch in range(args.epochs):
        for batch, (Q, D) in enumerate(dataloader):
            optimizer.zero_grad()

            query_embs, doc_embs = model(list(Q), list(D))
            loss, c_loss, d_loss = model.loss(query_embs, doc_embs, beta=args.beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            wandb.log({
                "loss/total": loss.item(),
                "loss/contrastive": c_loss.item(),
                "loss/distillation": d_loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }, step=step)

            if batch % 10 == 0:
                print(
                    f"Epoch {epoch} Batch {batch} | "
                    f"Loss: {loss.item():.4f} "
                    f"CL: {c_loss.item():.4f} "
                    f"DL: {d_loss.item():.4f}"
                )

            if step % args.ckpt_interval == 0 and step > 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                }, "checkpoints/ckpt_step.pt")
                print(f"Checkpoint saved at step {step}")

            step += 1

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bi-encoder model")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.0, help="Distillation loss weight")
    parser.add_argument("--min_rel", type=int, default=3, help="Minimum relevance score for ANTIQUE TRECDataset")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_interval", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--output", type=str, default="bi_encoder.pth", help="Output model path")
    parser.add_argument("--wandb_project", type=str, default="self-dist-retrieval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TRECDataset(min_rel=args.min_rel)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = BiEncoder(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=len(loader) * args.epochs
    )

    wandb.init(project=args.wandb_project, config=vars(args))

    model = train(loader, model, optimizer, scheduler, args, device=device)
    torch.save(model.state_dict(), args.output)
    wandb.finish()
