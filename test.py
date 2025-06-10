# test.py
import os
import numpy as np
import torch
import pretty_midi
from models.models import Generator

# --- Параметры ---
CHECKPOINT_PATH = 'checkpoints/G_AB_epoch_30.pth'  # Укажите путь к вашему генератору
INPUT_FILE_PATH = 'data/CP_C/train/имя_вашего_файла.npy'  # Укажите путь к .npy файлу для теста
OUTPUT_DIR = 'output/midi'


def save_piano_roll_to_midi(piano_roll, filename, fs=16, program=0):
    """
    Сохраняет piano roll (время, питч) в MIDI файл.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Добавляем паддинг для питчей, чтобы получить 128
    padded_piano_roll = np.zeros((piano_roll.shape[0], 128))
    padded_piano_roll[:, 24:24 + 84] = piano_roll

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=program)

    # Конвертация piano roll в ноты
    notes = pretty_midi.piano_roll_to_notes(padded_piano_roll, fs=fs, program=program)
    instrument.notes.extend(notes)

    midi.instruments.append(instrument)
    midi.write(filename)
    print(f"MIDI файл сохранен в {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели генератора
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    generator.eval()

    # Загрузка входных данных
    input_pr = np.load(INPUT_FILE_PATH)
    input_tensor = torch.from_numpy(input_pr).unsqueeze(0).to(device)  # Добавляем batch-измерение

    # Генерация
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Преобразование в numpy
    output_pr = output_tensor.cpu().numpy().squeeze()

    # Бинаризация результата
    output_pr[output_pr >= 0.5] = 1
    output_pr[output_pr < 0.5] = 0

    # Сохранение результата в MIDI
    output_filename = os.path.join(OUTPUT_DIR, f"generated_{os.path.basename(INPUT_FILE_PATH)}.mid")
    save_piano_roll_to_midi(output_pr, output_filename)

