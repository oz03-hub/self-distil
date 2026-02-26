import os
import torch
from torch.utils.data import DataLoader
import wandb

from util import TRECDataset
from bi_encoder import BiEncoder

from dotenv import load_dotenv
load_dotenv()


def train(
    dataloader: DataLoader, model: BiEncoder, optimizer, scheduler, epochs=1, device="cpu"
):
    model.train()

    print(len(dataloader))

    step = 0
    for epoch in range(epochs):
        for batch, (Q, D) in enumerate(dataloader):
            optimizer.zero_grad()

            query_embs, doc_embs = model(list(Q), list(D))
            loss, c_loss, d_loss = model.loss(query_embs, doc_embs)

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

            if step % 200 == 0 and step > 0:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TRECDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    model = BiEncoder("BAAI/bge-base-en-v1.5").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=len(loader) * 5
    )

    wandb.init(project="self-dist-retrieval", config={
        "model": "BAAI/bge-base-en-v1.5",
        "batch_size": 64,
        "lr": 2e-5,
        "epochs": 5,
        "temperature": 0.05,
    })

    model = train(loader, model, optimizer, scheduler, epochs=5, device=device)
    torch.save(model.state_dict(), "bi_encoder.pth")
    wandb.finish()
