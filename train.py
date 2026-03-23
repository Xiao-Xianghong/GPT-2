import torch
import os

# --- PyTorch CUDA Diagnostics ---
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Training on CPU.")
# --------------------------------

from torch.utils.data import DataLoader, random_split
from model import GPT, GPTConfig
from data import MyDataset

# --- Configuration ---
# Create a directory for checkpoints if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# --- Model and Optimizer Setup ---
config = GPTConfig()
model = GPT(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# --- Data Loading ---
dataset = MyDataset('input.txt') #[n_chunks] -> (x, y)

dataset_size = len(dataset)
val_size = int(0.1 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) #取出batch数个[4, (x, y)]
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) ##取出batch数个[4, (x, y)]
 
# --- Training and Evaluation Functions ---
def train_epoch(model, optimizer, scheduler, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, targets=y)

        optimizer.zero_grad() #无需传参！调用F.cross_entropy时已经确定loss在哪被算出来并可以反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
    return total_loss / len(train_loader) #整个 epoch 中，平均每个批次 (batch) 的损失值

def eval_epoch(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss / len(val_loader) #整个 epoch 中，平均每个批次 (batch) 的损失值

# --- Main Training Loop ---
for epoch in range(500):
    train_loss = train_epoch(model, optimizer, scheduler, train_loader, device)
    val_loss = eval_epoch(model, val_loader, device)
    
    print(f"Epoch: {epoch + 1}/500, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save checkpoint every 250 epochs
    if (epoch + 1) % 250 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }
        # Save the model checkpoint
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch + 1}.pt')
        print(f"--- Checkpoint saved at epoch {epoch + 1} ---")


