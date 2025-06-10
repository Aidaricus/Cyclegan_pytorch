import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from models.classifier import GenreClassifier, ClassifierLoss
from data.dataset import ClassifierDataset
from config import Config
from utils.visualization import plot_training_curves
from utils.metrics import calculate_accuracy, calculate_confusion_matrix


class ClassifierTrainer:
    """Тренер для жанрового классификатора"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Модель и оптимизатор
        self.model = GenreClassifier(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            ndf=config.ndf
        ).to(self.device)

        self.criterion = ClassifierLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Планировщик learning rate
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.epoch_step,
            gamma=config.lr_decay
        )

        # Датасеты
        self.train_dataset = ClassifierDataset(
            dataset_A_dir=config.dataset_A_dir,
            dataset_B_dir=config.dataset_B_dir,
            phase='train',
            normalize=True
        )

        self.test_dataset = ClassifierDataset(
            dataset_A_dir=config.dataset_A_dir,
            dataset_B_dir=config.dataset_B_dir,
            phase='test',
            normalize=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Для логирования
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        # Создание директорий
        self.checkpoint_dir = os.path.join(
            config.checkpoint_dir,
            f"classifier_{config.dataset_A_dir}2{config.dataset_B_dir}",
            datetime.now().strftime('%Y-%m-%d'),
            f"sigma_{config.sigma_c}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.log_dir = os.path.join(
            config.log_dir,
            f"classifier_{config.dataset_A_dir}2{config.dataset_B_dir}",
            datetime.now().strftime('%Y-%m-%d'),
            f"sigma_{config.sigma_c}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """Обучение одной эпохи"""
        self.model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Обнуление градиентов
            self.optimizer.zero_grad()

            # Прямой проход
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # Обратный проход
            loss.backward()
            self.optimizer.step()

            # Подсчет метрик
            batch_loss = loss.item()
            batch_accuracy = calculate_accuracy(outputs, labels)

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

            # Обновление прогресс-бара
            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_accuracy:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Средние значения за эпоху
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)

        return avg_loss, avg_accuracy

    def validate_epoch(self, epoch):
        """Валидация"""
        self.model.eval()

        epoch_accuracy = 0.0
        num_batches = len(self.test_loader)
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc='Validation'):
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Тестируем робастность с разными уровнями шума
                if hasattr(self.config, 'test_with_noise') and self.config.test_with_noise:
                    outputs = self.model(data, add_noise=True, noise_std=self.config.sigma_c)
                else:
                    outputs = self.model(data)

                batch_accuracy = calculate_accuracy(outputs, labels)
                epoch_accuracy += batch_accuracy

                # Сохраняем для confusion matrix
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(
                    torch.argmax(labels, dim=1).cpu().numpy() if labels.dim() > 1 else labels.cpu().numpy())

        avg_accuracy = epoch_accuracy / num_batches
        self.test_accuracies.append(avg_accuracy)

        # Вычисляем confusion matrix
        cm = calculate_confusion_matrix(all_predictions, all_labels)

        print(f'Validation Accuracy: {avg_accuracy:.4f}')
        print(f'Confusion Matrix:\n{cm}')

        return avg_accuracy, cm

    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Сохранение checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'config': self.config
        }

        # Сохраняем последний checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Сохраняем лучший checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)

        print(f'Checkpoint saved: {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """Загрузка checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.test_accuracies = checkpoint.get('test_accuracies', [])

        print(f'Checkpoint loaded: {checkpoint_path}')
        return checkpoint['epoch'], checkpoint['accuracy']

    def train(self):
        """Основной цикл обучения"""
        print("Starting classifier training...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

        best_accuracy = 0.0
        start_epoch = 0

        # Загружаем checkpoint если нужно
        if self.config.continue_train:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
            if os.path.exists(checkpoint_path):
                start_epoch, _ = self.load_checkpoint(checkpoint_path)
                start_epoch += 1

        for epoch in range(start_epoch, self.config.epochs):
            print(f'\nEpoch {epoch}/{self.config.epochs - 1}')
            print('-' * 60)

            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)

            # Валидация
            val_acc, confusion_matrix = self.validate_epoch(epoch)

            # Обновление learning rate
            self.scheduler.step()

            # Сохранение checkpoint
            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc

            self.save_checkpoint(epoch, val_acc, is_best)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Acc: {val_acc:.4f} (Best: {best_accuracy:.4f})')

            # Сохраняем графики обучения каждые 10 эпох
            if (epoch + 1) % 10 == 0:
                self.save_training_plots()

        print(f'\nTraining completed! Best accuracy: {best_accuracy:.4f}')
        self.save_training_plots()

    def save_training_plots(self):
        """Сохранение графиков обучения"""
        plot_training_curves(
            train_losses=self.train_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.test_accuracies,
            save_path=os.path.join(self.log_dir, 'training_curves.png')
        )


def main():
    parser = argparse.ArgumentParser(description='Train Genre Classifier')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--dataset_A_dir', type=str, required=True, help='Dataset A directory')
    parser.add_argument('--dataset_B_dir', type=str, required=True, help='Dataset B directory')
    parser.add_argument('--sigma_c', type=float, default=1.0, help='Noise std for classifier')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--continue_train', action='store_true', help='Continue training')

    args = parser.parse_args()

    # Загружаем конфиг и обновляем аргументами
    config = Config.from_file(args.config)
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    # Обучение
    trainer = ClassifierTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
