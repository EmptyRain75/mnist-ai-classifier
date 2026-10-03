"""Train and evaluate the basic MNIST ViT: python src/06_vit.py."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ----------------------------
# Data
# ----------------------------

def load_mnist(batch_size=32, val_size=10000, seed=42):
    """Create seeded MNIST train/validation/test DataLoaders."""
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_dataset, val_dataset = random_split(
        full_train,
        [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ----------------------------
# Model
# ----------------------------

def extract_patches(images, patch_size=7):
    """Convert (B, C, H, W) images into (B, N, C*P*P) patch vectors."""
    batch_size, channels, height, width = images.shape
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size.")

    patches = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(
        batch_size,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


class PatchEmbedding(nn.Module):
    """Apply one shared linear projection to every flattened patch."""

    def __init__(self, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim
        )

    def forward(self, images):
        return self.projection(extract_patches(images, self.patch_size))


class TokenPreparation(nn.Module):
    """Prepend a shared CLS token and add learned positional embeddings."""

    def __init__(self, num_patches=16, embed_dim=64):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, patch_embeddings):
        batch_size, num_patches, embed_dim = patch_embeddings.shape
        if num_patches != self.num_patches or embed_dim != self.embed_dim:
            raise ValueError("Unexpected patch embedding shape.")
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeddings], dim=1)
        return tokens + self.position_embeddings


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=128):
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by positive num_heads.")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, tokens):
        normalized_tokens = self.norm1(tokens)
        attention_output, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        tokens = tokens + attention_output
        return tokens + self.ffn(self.norm2(tokens))


class ViT(nn.Module):
    """Small MNIST ViT with two encoder blocks and a CLS classifier."""

    def __init__(
        self,
        image_size=28,
        patch_size=7,
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        mlp_dim=128,
        num_layers=2,
        num_classes=10,
    ):
        super().__init__()
        if patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must be positive and divide image_size.")

        self.image_size = image_size
        self.in_channels = in_channels
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.token_preparation = TokenPreparation(num_patches, embed_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        expected_shape = (self.in_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected images shaped (B, {expected_shape}), "
                f"got {tuple(images.shape)}."
            )

        tokens = self.token_preparation(self.patch_embedding(images))
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return self.head(tokens[:, 0])


# ----------------------------
# Training + Evaluation
# ----------------------------

def evaluate(model, data_loader, criterion, device):
    """Compute sample-weighted mean loss and classification accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
    return {"loss": total_loss / total, "accuracy": correct / total}


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, n_epochs=10
):
    """Train for fixed epochs; return runtime including per-epoch validation."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train loss: {total_loss / total:.4f} | "
            f"Train acc: {correct / total:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start_time


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# One-click Run
# ----------------------------

def main():
    batch_size = 32
    learning_rate = 0.001
    n_epochs = 10
    seed = 42

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_mnist(
        batch_size=batch_size, seed=seed
    )
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )

    print("Training ViT...")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"Device           : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(device)}")
    print(f"Seed             : {seed}")
    print("Optimizer        : Adam")
    print(f"Learning rate    : {learning_rate}")
    print(f"Epochs           : {n_epochs}")
    print(f"Batch size       : {batch_size}")
    print(f"Trainable params : {count_parameters(model):,}\n")

    training_run_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, n_epochs
    )
    train_eval_loader = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=False
    )

    print("\nFinal Results")
    print("-------------")
    for name, loader in {
        "Training": train_eval_loader,
        "Validation": val_loader,
        "Test": test_loader,
    }.items():
        metrics = evaluate(model, loader, criterion, device)
        print(
            f"{name:<10} | Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    print(f"\nTraining run time: {training_run_time:.2f} s")
    print("Runtime includes per-epoch validation; excludes final evaluation.")


if __name__ == "__main__":
    main()
