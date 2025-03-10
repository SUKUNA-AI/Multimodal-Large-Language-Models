from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(model_path: str, device: str) -> Tuple['PaliGemmaForConditionalGeneration', AutoTokenizer]:
    """
    Загружает модель PaliGemma и соответствующий токенизатор из указанного пути.

    Args:
        model_path (str): Путь к директории с файлами модели (веса, конфигурация, токенизатор).
        device (str): Устройство, на которое будет загружена модель ('cpu', 'cuda', 'mps').

    Returns:
        Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]: Кортеж из модели и токенизатора.
    """
    # Загрузка токенизатора из указанного пути
    # padding_side="right" указывает, что заполнение добавляется справа
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    # Проверка, что токенизатор настроен на заполнение справа (важно для генеративных моделей)
    assert tokenizer.padding_side == "right", "Tokenizer must pad on the right side"

    # Поиск всех файлов с расширением .safetensors в директории модели
    # Эти файлы содержат веса модели в формате SafeTensors
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Инициализация словаря для хранения тензоров (весов модели)
    tensors = {}
    # Загрузка всех .safetensors файлов по очереди
    for safetensors_file in safetensors_files:
        # Открытие файла SafeTensors с использованием PyTorch и загрузка на CPU
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            # Перебор всех ключей (названий тензоров) в файле
            for key in f.keys():
                # Добавление тензора в словарь по соответствующему ключу
                tensors[key] = f.get_tensor(key)

    # Загрузка конфигурации модели из файла config.json
    with open(os.path.join(model_path, "config.json"), "r") as f:
        # Чтение JSON-файла и преобразование в словарь
        model_config_file = json.load(f)
        # Создание объекта конфигурации PaliGemma на основе данных из файла
        config = PaliGemmaConfig(**model_config_file)

    # Создание экземпляра модели PaliGemma с использованием конфигурации
    # Модель сразу перемещается на указанное устройство
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Загрузка весов модели из словаря тензоров
    # strict=False позволяет игнорировать отсутствующие или лишние ключи
    model.load_state_dict(tensors, strict=False)

    # Привязка весов головы модели к весам эмбеддингов для оптимизации памяти
    model.tie_weights()

    # Возвращаем кортеж из модели и токенизатора
    return model, tokenizer
