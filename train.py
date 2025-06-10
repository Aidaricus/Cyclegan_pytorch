# train.py
import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.models import Generator, Discriminator
from tqdm import tqdm

# --- Параметры ---
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.0002
B1 = 0.5
B2 = 0.999
LAMBDA_CYCLE = 10.0
DATA_DIR_A = 'data/CP_C/train_beta'
DATA_DIR_B = 'data/CP_P/train_beta'
CHECKPOINT_DIR = 'checkpoints'
INPUT_SHAPE = (1, 64, 84)  # (Каналы, Время, Питчи)


# --- Класс датасета ---
class PianoRollDataset(Dataset):
    def __init__(self, dir_a, dir_b):
        self.files_a = [os.path.join(dir_a, f) for f in os.listdir(dir_a)]
        self.files_b = [os.path.join(dir_b, f) for f in os.listdir(dir_b)]

    def __getitem__(self, index):
        item_a = np.load(self.files_a[index % len(self.files_a)])
        item_b = np.load(self.files_b[np.random.randint(0, len(self.files_b) - 1)])
        item_a = item_a.transpose(2, 0, 1)  # теперь (1, 64, 84)
        item_b = item_b.transpose(2, 0, 1)

        assert item_a.shape == (1, 64, 84), f"Неожиданная форма: {item_a.shape}"
        assert item_b.shape == (1, 64, 84), f"Неожиданная форма: {item_b.shape}"
        # print(f"Item A shape: {item_a.shape}, Item B shape: {item_b.shape}")
        return torch.from_numpy(item_a).float(), torch.from_numpy(item_b).float()

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))


# --- Основной скрипт ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Инициализация моделей
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Инициализация лоссов
    criterion_gan = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)

    # Инициализация оптимизаторов
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(B1, B2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(B1, B2))

    # Загрузка данных
    dataset = PianoRollDataset(DATA_DIR_A, DATA_DIR_B)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Цикл обучения ---
    for epoch in range(EPOCHS):
        for i, (real_A, real_B) in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Целевые метки для adversarial loss
            valid = torch.ones(real_A.size(0), 1, 15, 20, device=device)  # Размер выхода дискриминатора
            fake = torch.zeros(real_A.size(0), 1, 15, 20, device=device)

            # --- Обучение генераторов ---
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # Генерация
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Adversarial loss
            loss_G_AB = criterion_gan(D_B(fake_B), valid)
            loss_G_BA = criterion_gan(D_A(fake_A), valid)
            loss_G = (loss_G_AB + loss_G_BA) / 2

            # Cycle loss
            cycle_A = G_BA(fake_B)
            cycle_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(cycle_A, real_A)
            loss_cycle_B = criterion_cycle(cycle_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Общий лосс для генераторов
            total_loss_G = loss_G + LAMBDA_CYCLE * loss_cycle
            total_loss_G.backward()
            optimizer_G.step()

            # --- Обучение дискриминатора A ---
            optimizer_D_A.zero_grad()

            loss_real = criterion_gan(D_A(real_A), valid)
            loss_fake = criterion_gan(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # --- Обучение дискриминатора B ---
            optimizer_D_B.zero_grad()

            loss_real = criterion_gan(D_B(real_B), valid)
            loss_fake = criterion_gan(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

        print(
            f"[Эпоха {epoch + 1}/{EPOCHS}] [D loss: {(loss_D_A + loss_D_B).item():.4f}] [G loss: {total_loss_G.item():.4f}, adv: {loss_G.item():.4f}, cycle: {loss_cycle.item():.4f}]")

        # Сохранение моделей
        if (epoch + 1) % 5 == 0:
            torch.save(G_AB.state_dict(), os.path.join(CHECKPOINT_DIR, f"G_AB_epoch_{epoch + 1}.pth"))
            torch.save(G_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f"G_BA_epoch_{epoch + 1}.pth"))
            print(f"Модели сохранены в эпоху {epoch + 1}")

