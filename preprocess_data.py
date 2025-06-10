# preprocess.py
import os
import numpy as np
import pretty_midi
from tqdm import tqdm

# --- Параметры ---
SOURCE_DIR_A = 'raw_data/A'
SOURCE_DIR_B = 'raw_data/B'
TARGET_DIR_A = 'data/A/train'
TARGET_DIR_B = 'data/B/train'
PITCH_RANGE = 84
TIME_STEPS_PER_BAR = 16
BARS_PER_PHRASE = 4
PHRASE_LENGTH = TIME_STEPS_PER_BAR * BARS_PER_PHRASE  # 64


def process_midi_file(filepath, phrase_length=64, time_steps_per_bar=16):
    """
    Загружает MIDI-файл, объединяет все треки и нарезает на фразы.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return []

    # Объединяем все инструменты в один piano roll
    # Устанавливаем fs (частоту дискретизации) равной time_steps_per_bar
    piano_roll = midi_data.get_piano_roll(fs=time_steps_per_bar)

    # Обрезаем диапазон питчей до 84 (C1 до C8, как в статье)
    # В pretty_midi нота C1 это 24
    piano_roll = piano_roll[24:24 + PITCH_RANGE, :]

    # Бинаризация: > 0 = нота есть, 0 = ноты нет
    piano_roll[piano_roll > 0] = 1

    # Транспонируем, чтобы получить (время, питч)
    piano_roll = piano_roll.T

    # Нарезаем на фразы
    num_phrases = piano_roll.shape[0] // phrase_length
    phrases = []
    for i in range(num_phrases):
        start = i * phrase_length
        end = start + phrase_length
        phrase = piano_roll[start:end, :]

        # Добавляем измерение канала (channel dimension)
        phrase = np.expand_dims(phrase, axis=0)  # Shape: (1, 64, 84)
        phrases.append(phrase)

    return phrases


def preprocess_domain(source_dir, target_dir):
    """
    Обрабатывает все MIDI-файлы в директории.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_count = 0
    for filename in tqdm(os.listdir(source_dir), desc=f"Обработка {source_dir}"):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            filepath = os.path.join(source_dir, filename)
            phrases = process_midi_file(filepath, phrase_length=PHRASE_LENGTH, time_steps_per_bar=TIME_STEPS_PER_BAR)

            for i, phrase in enumerate(phrases):
                # Сохраняем каждую фразу как отдельный .npy файл
                save_path = os.path.join(target_dir, f"{os.path.splitext(filename)[0]}_{i}.npy")
                np.save(save_path, phrase.astype(np.float32))
                file_count += 1

    print(f"Создано {file_count} файлов в {target_dir}")


if __name__ == '__main__':
    print("Начало предобработки данных...")
    preprocess_domain(SOURCE_DIR_A, TARGET_DIR_A)
    preprocess_domain(SOURCE_DIR_B, TARGET_DIR_B)
    print("Предобработка завершена.")

