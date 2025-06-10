import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from pathlib import Path

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MusicVisualization:
    """Класс для визуализации музыкальных данных"""

    def __init__(self, config=None):
        self.config = config
        self.pitch_names = self._get_pitch_names()

    def _get_pitch_names(self):
        """Генерация названий нот"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octaves = range(-1, 10)  # C-1 to G9

        pitch_names = []
        for octave in octaves:
            for note in notes:
                pitch_names.append(f"{note}{octave}")

        return pitch_names

    def plot_piano_roll(self, piano_roll, title="Piano Roll",
                        save_path=None, figsize=(15, 8),
                        pitch_start=24, pitch_end=108):
        """
        Визуализация piano roll

        Args:
            piano_roll: numpy array [time_steps, pitches] или [batch, time_steps, pitches]
            title: заголовок графика
            save_path: путь для сохранения
            figsize: размер фигуры
            pitch_start: начальная нота (MIDI number)
            pitch_end: конечная нота (MIDI number)
        """
        if piano_roll.ndim == 3:
            piano_roll = piano_roll[0]  # Берем первый семпл из батча

        fig, ax = plt.subplots(figsize=figsize)

        # Отображаем piano roll
        im = ax.imshow(piano_roll.T, aspect='auto', origin='lower',
                       cmap='Blues', interpolation='nearest')

        # Настройка осей
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('MIDI Pitch')
        ax.set_title(title)

        # Подписи для pitch axis
        pitch_range = pitch_end - pitch_start
        if pitch_range <= 84:  # Стандартный диапазон
            # Показываем каждую октаву
            octave_ticks = []
            octave_labels = []
            for octave in range(pitch_start // 12, (pitch_end // 12) + 1):
                c_note = octave * 12  # C ноты
                if pitch_start <= c_note < pitch_end:
                    octave_ticks.append(c_note - pitch_start)
                    octave_labels.append(f"C{octave - 1}")

            ax.set_yticks(octave_ticks)
            ax.set_yticklabels(octave_labels)

        # Добавляем colorbar
        plt.colorbar(im, ax=ax, label='Note Velocity')

        # Сетка
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Piano roll saved: {save_path}")

        return fig, ax

    def plot_transfer_comparison(self, original, transferred, cycle=None,
                                 titles=None, save_path=None, figsize=(20, 6)):
        """
        Сравнение оригинала, переноса и цикла

        Args:
            original: оригинальный piano roll
            transferred: перенесенный piano roll
            cycle: циклический piano roll (опционально)
            titles: заголовки для подграфиков
            save_path: путь для сохранения
        """
        num_plots = 3 if cycle is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)

        if titles is None:
            titles = ['Original', 'Transferred', 'Cycle']

        piano_rolls = [original, transferred]
        if cycle is not None:
            piano_rolls.append(cycle)

        for i, (piano_roll, title) in enumerate(zip(piano_rolls, titles)):
            if piano_roll.ndim == 3:
                piano_roll = piano_roll[0]

            im = axes[i].imshow(piano_roll.T, aspect='auto', origin='lower',
                                cmap='Blues', interpolation='nearest')
            axes[i].set_title(title)
            axes[i].set_xlabel('Time Steps')

            if i == 0:
                axes[i].set_ylabel('MIDI Pitch')

            # Colorbar только для последнего графика
            if i == len(piano_rolls) - 1:
                plt.colorbar(im, ax=axes[i], label='Note Velocity')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transfer comparison saved: {save_path}")

        return fig, axes

    def plot_training_curves(self, train_losses=None, train_accuracies=None,
                             val_accuracies=None, save_path=None, figsize=(15, 5)):
        """
        Визуализация кривых обучения

        Args:
            train_losses: потери на обучении
            train_accuracies: точность на обучении
            val_accuracies: точность на валидации
            save_path: путь для сохранения
        """
        num_plots = sum([train_losses is not None,
                         train_accuracies is not None and val_accuracies is not None])

        if num_plots == 0:
            print("No data to plot")
            return None, None

        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # График потерь
        if train_losses is not None:
            epochs = range(1, len(train_losses) + 1)
            axes[plot_idx].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Loss')
            axes[plot_idx].set_title('Training Loss')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # График точности
        if train_accuracies is not None and val_accuracies is not None:
            epochs = range(1, len(train_accuracies) + 1)
            axes[plot_idx].plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
            axes[plot_idx].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].set_title('Model Accuracy')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved: {save_path}")

        return fig, axes

    def plot_cyclegan_losses(self, losses_dict, save_path=None, figsize=(15, 10)):
        """
        Визуализация потерь CycleGAN

        Args:
            losses_dict: словарь с потерями {'G_A2B': [...], 'G_B2A': [...], ...}
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Потери генераторов
        if 'G_A2B' in losses_dict and 'G_B2A' in losses_dict:
            epochs = range(1, len(losses_dict['G_A2B']) + 1)
            axes[0].plot(epochs, losses_dict['G_A2B'], label='G A→B', linewidth=2)
            axes[0].plot(epochs, losses_dict['G_B2A'], label='G B→A', linewidth=2)
            axes[0].set_title('Generator Losses')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Потери дискриминаторов
        if 'D_A' in losses_dict and 'D_B' in losses_dict:
            epochs = range(1, len(losses_dict['D_A']) + 1)
            axes[1].plot(epochs, losses_dict['D_A'], label='D A', linewidth=2)
            axes[1].plot(epochs, losses_dict['D_B'], label='D B', linewidth=2)
            axes[1].set_title('Discriminator Losses')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Cycle loss
        if 'cycle_loss' in losses_dict:
            epochs = range(1, len(losses_dict['cycle_loss']) + 1)
            axes[2].plot(epochs, losses_dict['cycle_loss'], 'purple', label='Cycle Loss', linewidth=2)
            axes[2].set_title('Cycle Consistency Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # Общие потери
        if 'G_total' in losses_dict and 'D_total' in losses_dict:
            epochs = range(1, len(losses_dict['G_total']) + 1)
            axes[3].plot(epochs, losses_dict['G_total'], label='Total Generator', linewidth=2)
            axes[3].plot(epochs, losses_dict['D_total'], label='Total Discriminator', linewidth=2)
            axes[3].set_title('Total Losses')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CycleGAN losses saved: {save_path}")

        return fig, axes

    def plot_classifier_robustness(self, robustness_results, save_path=None, figsize=(10, 6)):
        """
        Визуализация робастности классификатора

        Args:
            robustness_results: словарь {noise_level: accuracy}
            save_path: путь для сохранения
        """
        noise_levels = sorted(robustness_results.keys())
        accuracies = [robustness_results[level] for level in noise_levels]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(noise_levels, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Noise Standard Deviation')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title('Classifier Robustness to Gaussian Noise')
        ax.grid(True, alpha=0.3)

        # Добавляем значения на точки
        for x, y in zip(noise_levels, accuracies):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness plot saved: {save_path}")

        return fig, ax

    def plot_transfer_statistics(self, results, save_path=None, figsize=(15, 10)):
        """
        Визуализация статистики переноса стиля

        Args:
            results: список результатов оценки переноса
            save_path: путь для сохранения
        """
        # Конвертируем в DataFrame для удобства
        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Гистограмма успешности переноса
        success_counts = df['is_successful'].value_counts()
        axes[0, 0].bar(['Failed', 'Successful'],
                       [success_counts.get(False, 0), success_counts.get(True, 0)],
                       color=['red', 'green'], alpha=0.7)
        axes[0, 0].set_title('Transfer Success Rate')
        axes[0, 0].set_ylabel('Number of Samples')

        # 2. Распределение силы переноса
        successful_df = df[df['is_successful'] == True]
        if not successful_df.empty:
            axes[0, 1].hist(successful_df['transfer_strength'], bins=20, alpha=0.7, color='blue')
            axes[0, 1].set_title('Distribution of Transfer Strength')
            axes[0, 1].set_xlabel('Transfer Strength')
            axes[0, 1].set_ylabel('Frequency')

        # 3. Scatter plot: Content preservation vs Transfer strength
        if not successful_df.empty:
            axes[1, 0].scatter(successful_df['content_diff'], successful_df['transfer_strength'],
                               alpha=0.6, color='purple')
            axes[1, 0].set_xlabel('Content Difference (MSE)')
            axes[1, 0].set_ylabel('Transfer Strength')
            axes[1, 0].set_title('Content Preservation vs Transfer Strength')

        # 4. Boxplot вероятностей
        prob_data = [df['source_prob'], df['transfer_prob'], df['cycle_prob']]
        axes[1, 1].boxplot(prob_data, labels=['Source', 'Transfer', 'Cycle'])
        axes[1, 1].set_title('Probability Distributions')
        axes[1, 1].set_ylabel('Probability')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transfer statistics saved: {save_path}")

        return fig, axes

    def create_interactive_piano_roll(self, piano_roll, title="Interactive Piano Roll",
                                      save_path=None):
        """
        Создание интерактивного piano roll с помощью Plotly

        Args:
            piano_roll: numpy array [time_steps, pitches]
            title: заголовок
            save_path: путь для сохранения HTML
        """
        if piano_roll.ndim == 3:
            piano_roll = piano_roll[0]

        # Создаем heatmap
        fig = go.Figure(data=go.Heatmap(
            z=piano_roll.T,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Time: %{x}<br>Pitch: %{y}<br>Velocity: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time Steps',
            yaxis_title='MIDI Pitch',
            width=1200,
            height=600
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive piano roll saved: {save_path}")

        return fig

    def create_genre_comparison_dashboard(self, genre_results, save_path=None):
        """
        Создание интерактивной панели сравнения жанров

        Args:
            genre_results: словарь {genre_pair: results}
            save_path: путь для сохранения HTML
        """
        # Подготавливаем данные
        comparison_data = []
        for genre_pair, results in genre_results.items():
            for result in results:
                comparison_data.append({
                    'genre_pair': genre_pair,
                    'transfer_strength': result['transfer_strength'],
                    'content_diff': result['content_diff'],
                    'is_successful': result['is_successful'],
                    'prob_diff': result['prob_diff']
                })

        df = pd.DataFrame(comparison_data)

        # Создаем subplot фигуру
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate by Genre Pair', 'Transfer Strength Distribution',
                            'Content vs Transfer Quality', 'Probability Differences'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "violin"}]]
        )

        # 1. Success rate by genre pair
        success_rates = df.groupby('genre_pair')['is_successful'].mean()
        fig.add_trace(
            go.Bar(x=success_rates.index, y=success_rates.values, name='Success Rate'),
            row=1, col=1
        )

        # 2. Transfer strength distribution
        for genre_pair in df['genre_pair'].unique():
            subset = df[(df['genre_pair'] == genre_pair) & (df['is_successful'] == True)]
            if not subset.empty:
                fig.add_trace(
                    go.Box(y=subset['transfer_strength'], name=genre_pair),
                    row=1, col=2
                )

        # 3. Content vs Transfer scatter
        successful_df = df[df['is_successful'] == True]
        for genre_pair in successful_df['genre_pair'].unique():
            subset = successful_df[successful_df['genre_pair'] == genre_pair]
            fig.add_trace(
                go.Scatter(
                    x=subset['content_diff'],
                    y=subset['transfer_strength'],
                    mode='markers',
                    name=genre_pair
                ),
                row=2, col=1
            )

        # 4. Probability differences
        for genre_pair in df['genre_pair'].unique():
            subset = df[df['genre_pair'] == genre_pair]
            fig.add_trace(
                go.Violin(y=subset['prob_diff'], name=genre_pair),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text="Genre Transfer Comparison Dashboard",
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Genre comparison dashboard saved: {save_path}")

        return fig


def plot_training_curves(train_losses=None, train_accuracies=None,
                         val_accuracies=None, save_path=None):
    """Wrapper функция для быстрого вызова"""
    viz = MusicVisualization()
    return viz.plot_training_curves(train_losses, train_accuracies, val_accuracies, save_path)


def plot_piano_roll(piano_roll, title="Piano Roll", save_path=None):
    """Wrapper функция для быстрого вызова"""
    viz = MusicVisualization()
    return viz.plot_piano_roll(piano_roll, title, save_path)


def plot_transfer_comparison(original, transferred, cycle=None, save_path=None):
    """Wrapper функция для быстрого вызова"""
    viz = MusicVisualization()
    return viz.plot_transfer_comparison(original, transferred, cycle, save_path=save_path)
