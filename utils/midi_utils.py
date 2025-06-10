import os
import numpy as np
import pretty_midi as pm
import copy
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings
from tqdm import tqdm

# Подавляем предупреждения pretty_midi
warnings.filterwarnings('ignore', category=UserWarning, module='pretty_midi')


class MIDIProcessor:
    """Класс для обработки MIDI файлов"""

    def __init__(self,
                 beat_resolution: int = 4,
                 bars_per_phrase: int = 4,
                 time_steps_per_bar: int = 16,
                 pitch_start: int = 24,  # C1
                 pitch_end: int = 108,  # C8
                 velocity_threshold: float = 0.0):

        self.beat_resolution = beat_resolution
        self.bars_per_phrase = bars_per_phrase
        self.time_steps_per_bar = time_steps_per_bar
        self.total_time_steps = bars_per_phrase * time_steps_per_bar
        self.pitch_start = pitch_start
        self.pitch_end = pitch_end
        self.pitch_range = pitch_end - pitch_start
        self.velocity_threshold = velocity_threshold

        # Названия нот для отладки
        self.note_names = self._generate_note_names()

    def _generate_note_names(self) -> List[str]:
        """Генерация названий нот"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_names = []

        for midi_num in range(128):
            octave = (midi_num // 12) - 1
            note = notes[midi_num % 12]
            note_names.append(f"{note}{octave}")

        return note_names

    def load_midi(self, midi_path: str) -> pm.PrettyMIDI:
        """Загружает MIDI файл с обработкой ошибок"""
        try:
            midi = pm.PrettyMIDI(midi_path)
            return midi
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {midi_path}: {str(e)}")

    def get_midi_info(self, midi: pm.PrettyMIDI) -> dict:
        """Получает информацию о MIDI файле"""
        # Время первого бита
        if midi.time_signature_changes:
            midi.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = midi.time_signature_changes[0].time
        else:
            first_beat_time = midi.estimate_beat_start()

        # Изменения темпа
        tempo_changes = midi.get_tempo_changes()

        # Тональность
        if len(midi.time_signature_changes) == 1:
            ts = midi.time_signature_changes[0]
            time_signature = f"{ts.numerator}/{ts.denominator}"
        else:
            time_signature = None

        # Длительность
        end_time = midi.get_end_time()

        # Инструменты
        instruments_info = []
        for i, instrument in enumerate(midi.instruments):
            instruments_info.append({
                'index': i,
                'name': instrument.name,
                'program': instrument.program,
                'is_drum': instrument.is_drum,
                'notes_count': len(instrument.notes)
            })

        return {
            'first_beat_time': first_beat_time,
            'num_time_signature_changes': len(midi.time_signature_changes),
            'time_signature': time_signature,
            'tempo': tempo_changes[1][0] if len(tempo_changes[1]) == 1 else None,
            'duration': end_time,
            'num_instruments': len(midi.instruments),
            'instruments': instruments_info,
            'total_notes': sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
        }

    def midi_filter(self, midi_info: dict) -> bool:
        """Фильтрует MIDI файлы по критериям качества"""
        # Проверка времени первого бита
        if midi_info['first_beat_time'] > 0.0:
            return False

        # Проверка изменений размера
        if midi_info['num_time_signature_changes'] > 1:
            return False

        # Проверка размера (только 4/4)
        if midi_info['time_signature'] not in ['4/4']:
            return False

        # Проверка длительности (минимум 4 такта)
        min_duration = 4 * (60.0 / 120.0) * 4  # 4 такта в 120 BPM
        if midi_info['duration'] < min_duration:
            return False

        # Проверка наличия нот
        if midi_info['total_notes'] < 10:
            return False

        return True

    def extract_piano_roll_from_midi(self, midi: pm.PrettyMIDI) -> Optional[np.ndarray]:
        """Извлекает piano roll из MIDI файла"""
        try:
            # Получаем piano roll с разрешением beat_resolution
            # Время на бит = 60 / (темп * beat_resolution)
            tempo = 120  # Используем стандартный темп если не указан
            if midi.get_tempo_changes()[1]:
                tempo = midi.get_tempo_changes()[1][0]

            # Время на семпл
            time_per_sample = 60.0 / (tempo * self.time_steps_per_bar)

            # Получаем piano roll
            piano_roll = midi.get_piano_roll(fs=1 / time_per_sample)

            # Транспонируем для формата [time, pitch]
            piano_roll = piano_roll.T

            return piano_roll

        except Exception as e:
            print(f"Error extracting piano roll: {str(e)}")
            return None

    def merge_instruments(self, midi: pm.PrettyMIDI, exclude_drums: bool = True) -> pm.PrettyMIDI:
        """Объединяет все инструменты в один трек"""
        merged_midi = pm.PrettyMIDI()
        merged_instrument = pm.Instrument(program=0, name='Merged')

        for instrument in midi.instruments:
            # Пропускаем ударные если указано
            if exclude_drums and instrument.is_drum:
                continue

            # Копируем все ноты
            for note in instrument.notes:
                # Ограничиваем диапазон нот
                if self.pitch_start <= note.pitch < self.pitch_end:
                    merged_instrument.notes.append(copy.deepcopy(note))

        # Сортируем ноты по времени начала
        merged_instrument.notes.sort(key=lambda x: x.start)

        merged_midi.instruments.append(merged_instrument)

        # Копируем метаданные
        merged_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        merged_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)

        return merged_midi

    def piano_roll_to_bars(self, piano_roll: np.ndarray, last_bar_mode: str = 'remove') -> Optional[np.ndarray]:
        """Конвертирует piano roll в такты"""
        if piano_roll.shape[0] == 0:
            return None

        # Обрезаем до нужного диапазона нот
        if piano_roll.shape[1] >= self.pitch_end:
            piano_roll = piano_roll[:, self.pitch_start:self.pitch_end]
        else:
            # Дополняем нулями если диапазон меньше
            padding_size = self.pitch_end - piano_roll.shape[1]
            padding = np.zeros((piano_roll.shape[0], padding_size))
            piano_roll = np.concatenate([piano_roll, padding], axis=1)
            piano_roll = piano_roll[:, self.pitch_start:self.pitch_end]

        # Проверяем, что диапазон нот правильный
        if piano_roll.shape[1] != self.pitch_range:
            print(f"Warning: Expected {self.pitch_range} pitches, got {piano_roll.shape[1]}")
            return None

        # Разбиваем на такты
        steps_per_bar = self.time_steps_per_bar
        total_steps = piano_roll.shape[0]

        # Обрабатываем неполные такты
        if total_steps % steps_per_bar != 0:
            remainder = total_steps % steps_per_bar

            if last_bar_mode == 'remove':
                # Удаляем неполный такт
                piano_roll = piano_roll[:-remainder]
            elif last_bar_mode == 'fill':
                # Дополняем нулями до полного такта
                padding_steps = steps_per_bar - remainder
                padding = np.zeros((padding_steps, piano_roll.shape[1]))
                piano_roll = np.concatenate([piano_roll, padding], axis=0)
            else:
                raise ValueError(f"Unknown last_bar_mode: {last_bar_mode}")

        if piano_roll.shape[0] == 0:
            return None

        # Разбиваем на такты
        num_bars = piano_roll.shape[0] // steps_per_bar
        bars = piano_roll.reshape(num_bars, steps_per_bar, self.pitch_range)

        return bars

    def bars_to_phrases(self, bars: np.ndarray, remove_empty: bool = True) -> np.ndarray:
        """Группирует такты в фразы"""
        if bars.shape[0] < self.bars_per_phrase:
            return np.array([])

        # Удаляем лишние такты
        num_complete_phrases = bars.shape[0] // self.bars_per_phrase
        bars = bars[:num_complete_phrases * self.bars_per_phrase]

        # Группируем в фразы
        phrases = bars.reshape(
            num_complete_phrases,
            self.total_time_steps,
            self.pitch_range
        )

        # Удаляем пустые фразы если указано
        if remove_empty:
            non_empty_mask = np.max(phrases, axis=(1, 2)) > self.velocity_threshold
            phrases = phrases[non_empty_mask]

        return phrases

    def process_midi_to_phrases(self, midi_path: str) -> Tuple[Optional[np.ndarray], str]:
        """Полная обработка MIDI файла в фразы"""
        try:
            # Загрузка MIDI
            midi = self.load_midi(midi_path)

            # Проверка качества
            midi_info = self.get_midi_info(midi)
            if not self.midi_filter(midi_info):
                return None, f"MIDI filter failed: {midi_info}"

            # Объединение инструментов
            merged_midi = self.merge_instruments(midi)

            # Извлечение piano roll
            piano_roll = self.extract_piano_roll_from_midi(merged_midi)
            if piano_roll is None:
                return None, "Failed to extract piano roll"

            # Конвертация в такты
            bars = self.piano_roll_to_bars(piano_roll)
            if bars is None:
                return None, "Failed to convert to bars"

            # Группировка в фразы
            phrases = self.bars_to_phrases(bars)
            if phrases.shape[0] == 0:
                return None, "No valid phrases found"

            # Бинаризация (убираем velocity)
            phrases_binary = (phrases > self.velocity_threshold).astype(np.float32)

            return phrases_binary, "Success"

        except Exception as e:
            return None, f"Processing error: {str(e)}"


# Функции для сохранения MIDI файлов

def piano_roll_to_pretty_midi(piano_roll: np.ndarray,
                              tempo: float = 120.0,
                              velocity: int = 100,
                              program: int = 0,
                              pitch_start: int = 24) -> pm.PrettyMIDI:
    """Конвертирует piano roll в PrettyMIDI объект"""

    # Создаем MIDI объект
    midi = pm.PrettyMIDI(initial_tempo=tempo)

    # Создаем инструмент
    instrument = pm.Instrument(program=program, name='Piano')

    # Параметры времени
    time_per_step = 60.0 / (tempo * 16)  # 16 шагов на такт
    threshold = time_per_step / 4  # Минимальная длительность ноты

    # Обрабатываем каждую ноту
    for pitch_idx in range(piano_roll.shape[1]):
        pitch = pitch_start + pitch_idx

        # Находим начала и концы нот
        note_events = np.diff(np.concatenate([[0], piano_roll[:, pitch_idx], [0]]).astype(int))
        note_starts = np.where(note_events == 1)[0]
        note_ends = np.where(note_events == -1)[0]

        # Создаем ноты
        for start_idx, end_idx in zip(note_starts, note_ends):
            start_time = start_idx * time_per_step
            end_time = end_idx * time_per_step

            # Проверяем минимальную длительность
            if end_time - start_time < threshold:
                end_time = start_time + threshold

            # Создаем ноту
            note = pm.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )

            instrument.notes.append(note)

    # Сортируем ноты по времени
    instrument.notes.sort(key=lambda x: x.start)

    midi.instruments.append(instrument)

    return midi


def save_midis(piano_rolls: np.ndarray,
               file_path: str,
               tempo: float = 120.0,
               velocity: int = 100,
               program: int = 0) -> None:
    """
    Сохраняет piano roll как MIDI файл

    Args:
        piano_rolls: numpy array формата [batch, time, pitch] или [batch, time, pitch, channels]
        file_path: путь для сохранения
        tempo: темп в BPM
        velocity: громкость нот (0-127)
        program: номер инструмента
    """

    # Обрабатываем размерности
    if piano_rolls.ndim == 4:
        # Убираем канальную размерность
        piano_rolls = piano_rolls.squeeze(-1)

    if piano_rolls.ndim == 3:
        # Берем первый семпл из батча
        piano_roll = piano_rolls[0]
    else:
        piano_roll = piano_rolls

    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Конвертируем и сохраняем
    midi = piano_roll_to_pretty_midi(
        piano_roll,
        tempo=tempo,
        velocity=velocity,
        program=program
    )

    midi.write(file_path)


def save_midis_with_probabilities(origin: np.ndarray,
                                  transfer: np.ndarray,
                                  cycle: np.ndarray,
                                  origin_probs: np.ndarray,
                                  transfer_probs: np.ndarray,
                                  cycle_probs: np.ndarray,
                                  sample_id: int,
                                  output_dir: str) -> None:
    """
    Сохраняет MIDI файлы с вероятностями в названиях

    Args:
        origin: оригинальный piano roll
        transfer: перенесенный piano roll
        cycle: циклический piano roll
        origin_probs: вероятности для оригинала [prob_A, prob_B]
        transfer_probs: вероятности для переноса
        cycle_probs: вероятности для цикла
        sample_id: идентификатор семпла
        output_dir: директория для сохранения
    """

    os.makedirs(output_dir, exist_ok=True)

    # Извлекаем вероятности
    prob_A_orig = origin_probs[0]
    prob_A_transfer = transfer_probs[0]
    prob_A_cycle = cycle_probs[0]

    # Создаем названия файлов с вероятностями
    origin_path = os.path.join(output_dir, f'{sample_id:03d}_origin_A{prob_A_orig:.3f}.mid')
    transfer_path = os.path.join(output_dir, f'{sample_id:03d}_transfer_A{prob_A_transfer:.3f}.mid')
    cycle_path = os.path.join(output_dir, f'{sample_id:03d}_cycle_A{prob_A_cycle:.3f}.mid')

    # Сохраняем файлы
    save_midis(origin, origin_path)
    save_midis(transfer, transfer_path)
    save_midis(cycle, cycle_path)


# Утилиты для анализа MIDI данных

def analyze_midi_dataset(midi_dir: str, output_file: str = None) -> dict:
    """Анализирует датасет MIDI файлов"""

    processor = MIDIProcessor()
    midi_files = list(Path(midi_dir).glob('*.mid')) + list(Path(midi_dir).glob('*.midi'))

    stats = {
        'total_files': len(midi_files),
        'valid_files': 0,
        'invalid_files': 0,
        'total_notes': 0,
        'total_duration': 0.0,
        'instruments_used': {},
        'time_signatures': {},
        'errors': [],
        'pitch_distribution': np.zeros(128),
        'duration_distribution': [],
        'note_density_distribution': []
    }

    print(f"Analyzing {len(midi_files)} MIDI files...")

    for midi_file in tqdm(midi_files):
        try:
            midi = processor.load_midi(str(midi_file))
            midi_info = processor.get_midi_info(midi)

            if processor.midi_filter(midi_info):
                stats['valid_files'] += 1
                stats['total_duration'] += midi_info['duration']
                stats['total_notes'] += midi_info['total_notes']

                # Статистика по тональности
                ts = midi_info['time_signature']
                stats['time_signatures'][ts] = stats['time_signatures'].get(ts, 0) + 1

                # Статистика по инструментам
                for inst_info in midi_info['instruments']:
                    if not inst_info['is_drum']:
                        prog = inst_info['program']
                        stats['instruments_used'][prog] = stats['instruments_used'].get(prog, 0) + 1

                # Анализ нот
                for instrument in midi.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            stats['pitch_distribution'][note.pitch] += 1

                # Плотность нот
                note_density = midi_info['total_notes'] / midi_info['duration']
                stats['note_density_distribution'].append(note_density)
                stats['duration_distribution'].append(midi_info['duration'])

            else:
                stats['invalid_files'] += 1

        except Exception as e:
            stats['invalid_files'] += 1
            stats['errors'].append(f"{midi_file.name}: {str(e)}")

    # Вычисляем статистики
    if stats['valid_files'] > 0:
        stats['avg_duration'] = stats['total_duration'] / stats['valid_files']
        stats['avg_notes_per_file'] = stats['total_notes'] / stats['valid_files']
        stats['avg_note_density'] = np.mean(stats['note_density_distribution'])

        # Наиболее используемые ноты
        pitch_counts = stats['pitch_distribution']
        most_used_pitches = np.argsort(pitch_counts)[-10:][::-1]
        stats['most_used_pitches'] = [
            (int(pitch), processor.note_names[pitch], int(pitch_counts[pitch]))
            for pitch in most_used_pitches if pitch_counts[pitch] > 0
        ]

    # Сохраняем отчет
    if output_file:
        import json
        with open(output_file, 'w') as f:
            # Конвертируем numpy arrays для JSON
            json_stats = copy.deepcopy(stats)
            json_stats['pitch_distribution'] = stats['pitch_distribution'].tolist()
            json.dump(json_stats, f, indent=2)

    return stats


def print_midi_analysis(stats: dict) -> None:
    """Выводит анализ MIDI датасета"""

    print("\n" + "=" * 60)
    print("MIDI DATASET ANALYSIS")
    print("=" * 60)

    print(f"Total files: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']} ({stats['valid_files'] / stats['total_files'] * 100:.1f}%)")
    print(f"Invalid files: {stats['invalid_files']}")

    if stats['valid_files'] > 0:
        print(f"\nDuration statistics:")
        print(f"  Total duration: {stats['total_duration']:.1f} seconds")
        print(f"  Average duration: {stats['avg_duration']:.1f} seconds")

        print(f"\nNote statistics:")
        print(f"  Total notes: {stats['total_notes']}")
        print(f"  Average notes per file: {stats['avg_notes_per_file']:.1f}")
        print(f"  Average note density: {stats['avg_note_density']:.2f} notes/sec")

        print(f"\nTime signatures:")
        for ts, count in sorted(stats['time_signatures'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {ts}: {count} files")

        print(f"\nMost used instruments (GM program numbers):")
        for prog, count in sorted(stats['instruments_used'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Program {prog}: {count} files")

        if 'most_used_pitches' in stats:
            print(f"\nMost used pitches:")
            for pitch, note_name, count in stats['most_used_pitches'][:5]:
                print(f"  {note_name} (MIDI {pitch}): {count} occurrences")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"  {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")


# Вспомогательные функции

def get_bar_piano_roll(piano_roll: np.ndarray,
                       time_steps_per_bar: int = 16,
                       last_bar_mode: str = 'remove') -> Optional[np.ndarray]:
    """
    Wrapper функция для обратной совместимости
    Конвертирует piano roll в format bars
    """
    processor = MIDIProcessor(time_steps_per_bar=time_steps_per_bar)
    return processor.piano_roll_to_bars(piano_roll, last_bar_mode)


def get_midi_info(midi: pm.PrettyMIDI) -> dict:
    """Wrapper функция для обратной совместимости"""
    processor = MIDIProcessor()
    return processor.get_midi_info(midi)


def midi_filter(midi_info: dict) -> bool:
    """Wrapper функция для обратной совместимости"""
    processor = MIDIProcessor()
    return processor.midi_filter(midi_info)


# Функции для визуализации MIDI данных

def plot_piano_roll_midi(piano_roll: np.ndarray,
                         title: str = "Piano Roll",
                         save_path: str = None,
                         pitch_start: int = 24,
                         pitch_end: int = 108) -> None:
    """Визуализация piano roll"""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 8))

    # Отображаем piano roll
    im = ax.imshow(piano_roll.T, aspect='auto', origin='lower',
                   cmap='Blues', interpolation='nearest')

    # Настройка осей
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title(title)

    # Подписи для pitch axis (каждая октава)
    octave_ticks = []
    octave_labels = []
    for octave in range(pitch_start // 12, (pitch_end // 12) + 1):
        c_note = octave * 12  # C ноты
        if pitch_start <= c_note < pitch_end:
            octave_ticks.append(c_note - pitch_start)
            octave_labels.append(f"C{octave - 1}")

    ax.set_yticks(octave_ticks)
    ax.set_yticklabels(octave_labels)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Note Velocity')

    # Сетка
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Piano roll visualization saved: {save_path}")

    plt.show()


def create_midi_statistics_plot(stats: dict, save_path: str = None) -> None:
    """Создает графики статистики MIDI данных"""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Распределение длительности
    if stats['duration_distribution']:
        axes[0, 0].hist(stats['duration_distribution'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Duration Distribution')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Распределение плотности нот
    if stats['note_density_distribution']:
        axes[0, 1].hist(stats['note_density_distribution'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Notes per Second')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Note Density Distribution')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Распределение высот нот
    if hasattr(stats['pitch_distribution'], '__len__'):
        pitch_dist = np.array(stats['pitch_distribution'])
        pitches = np.arange(len(pitch_dist))

        # Показываем только используемые ноты
        used_pitches = pitches[pitch_dist > 0]
        used_counts = pitch_dist[pitch_dist > 0]

        axes[1, 0].bar(used_pitches, used_counts, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('MIDI Pitch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pitch Distribution')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Статистика по тональностям
    if stats['time_signatures']:
        ts_names = list(stats['time_signatures'].keys())
        ts_counts = list(stats['time_signatures'].values())

        axes[1, 1].bar(ts_names, ts_counts, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Time Signature')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Time Signature Distribution')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved: {save_path}")

    plt.show()


# Функции для валидации и тестирования

def validate_midi_processing(input_dir: str, output_dir: str, num_samples: int = 5) -> None:
    """Валидирует процесс обработки MIDI файлов"""

    processor = MIDIProcessor()
    midi_files = list(Path(input_dir).glob('*.mid'))[:num_samples]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Validating MIDI processing on {len(midi_files)} files...")

    for i, midi_file in enumerate(midi_files):
        print(f"\nProcessing: {midi_file.name}")

        # Обработка
        phrases, status = processor.process_midi_to_phrases(str(midi_file))

        if phrases is not None:
            print(f"  Success: {phrases.shape[0]} phrases generated")

            # Сохраняем первую фразу как пример
            if phrases.shape[0] > 0:
                example_phrase = phrases[0]

                # Сохраняем как MIDI
                output_path = os.path.join(output_dir, f'processed_{i}_{midi_file.stem}.mid')
                save_midis(example_phrase, output_path)

                # Сохраняем визуализацию
                plot_path = os.path.join(output_dir, f'processed_{i}_{midi_file.stem}.png')
                plot_piano_roll_midi(example_phrase,
                                     title=f"Processed: {midi_file.name}",
                                     save_path=plot_path)
        else:
            print(f"  Failed: {status}")


if __name__ == "__main__":
    # Пример использования
    import argparse

    parser = argparse.ArgumentParser(description='MIDI Utilities')
    parser.add_argument('--analyze', type=str, help='Analyze MIDI directory')
    parser.add_argument('--validate', nargs=2, metavar=('INPUT_DIR', 'OUTPUT_DIR'),
                        help='Validate MIDI processing')
    parser.add_argument('--convert', nargs=2, metavar=('INPUT_FILE', 'OUTPUT_FILE'),
                        help='Convert single MIDI file')

    args = parser.parse_args()

    if args.analyze:
        print(f"Analyzing MIDI files in: {args.analyze}")
        stats = analyze_midi_dataset(args.analyze, f"{args.analyze}_analysis.json")
        print_midi_analysis(stats)
        create_midi_statistics_plot(stats, f"{args.analyze}_statistics.png")

    elif args.validate:
        input_dir, output_dir = args.validate
        validate_midi_processing(input_dir, output_dir)

    elif args.convert:
        input_file, output_file = args.convert
        processor = MIDIProcessor()
        phrases, status = processor.process_midi_to_phrases(input_file)

        if phrases is not None:
            print(f"Converted: {phrases.shape[0]} phrases")
            if phrases.shape[0] > 0:
                save_midis(phrases[0], output_file)
                print(f"Saved first phrase to: {output_file}")
        else:
            print(f"Conversion failed: {status}")

    else:
        print("Use --help for usage information")
