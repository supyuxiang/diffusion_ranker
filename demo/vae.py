import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os


# =========================
# 超参数
# =========================
BATCH_SIZE = 16
INPUT_DIM = 784
HIDDEN_DIM = 400
LATENT_DIM = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建输出目录
os.makedirs("results", exist_ok=True)


# =========================
# 数据集
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================
# VAE模型
# =========================
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

        # Decoder
        self.fc2 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, INPUT_DIM)
        self._init_weights()
        
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(h))
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# =========================
# Loss函数
# =========================
def loss_function(x_hat, x, mu, logvar):
    # Reconstruction Loss
    recon_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='sum'
    )

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss


# =========================
# 初始化模型
# =========================
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# 训练函数
# =========================
def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, INPUT_DIM).to(DEVICE)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(x)

            loss = loss_function(x_hat, x, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")


# =========================
# 重建图像
# =========================
def reconstruct():
    model.eval()

    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.view(-1, INPUT_DIM).to(DEVICE)

        x_hat, _, _ = model(x)

        x_hat = x_hat.view(-1, 1, 28, 28)

        save_image(
            x_hat[:64],
            "results/reconstruction.png",
            nrow=8
        )

    print("Reconstructed images saved.")


# =========================
# 随机生成图像
# =========================
def generate():
    model.eval()

    with torch.no_grad():
        z = torch.randn(64, LATENT_DIM).to(DEVICE)

        samples = model.decode(z)

        samples = samples.view(-1, 1, 28, 28)

        save_image(
            samples,
            "results/generated.png",
            nrow=8
        )

    print("Generated images saved.")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    train()
    reconstruct()
    generate()

    torch.save(model.state_dict(), "results/vae.pth")

    print("Model saved.")