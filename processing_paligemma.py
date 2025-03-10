from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Средние значения для нормализации ImageNet (RGB каналы)
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]

# Стандартное отклонение для нормализации ImageNet (RGB каналы)
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


# Функция для добавления токенов изображения в начало запроса
def add_image_tokens_to_prompt(prefix_propmt, bos_token, image_seq_len, image_token):
    # Создает строку, где сначала идут токены изображения (image_token), повторенные image_seq_len раз,
    # затем добавляется начальный токен (bos_token) и исходный запрос (prefix_propmt)
    return f'{image_token * image_seq_len}{bos_token}{prefix_propmt}\n'


# Функция изменения размера изображения
def resize(
        image: Image,                    # Входное изображение в формате PIL Image
        size: Tuple[int, int],          # Кортеж с новой высотой и шириной (height, width)
        resample: Image.Resampling = None,  # Метод интерполяции для изменения размера
        reducing_gap: Optional[int] = None, # Опциональный параметр для оптимизации уменьшения
) -> np.ndarray:                        # Возвращает массив numpy с измененным изображением
    height, width = size                # Распаковка размеров из кортежа
    # Изменение размера изображения с использованием заданных параметров
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image                # Возвращает изображение с новым размером


# Функция масштабирования значений изображения
def rescale(image: np.ndarray,         # Входной массив изображения
            scale: float,              # Коэффициент масштабирования
            dtype: np.dtype = np.float32) -> np.ndarray:  # Желаемый тип данных после масштабирования
    rescaled_image = image * scale     # Умножение всех значений изображения на масштаб
    rescaled_image = rescaled_image.astype(dtype)  # Приведение результата к заданному типу данных
    return rescaled_image              # Возвращает масштабированное изображение


# Функция нормализации изображения
def normalize(
        image: np.ndarray,             # Входной массив изображения
        mean: Union[float, Iterable[float]] = None,  # Среднее значение для нормализации
        std: Union[float, Iterable[float]] = None,   # Стандартное отклонение для нормализации
        dtype: np.dtype = np.float32,  # Желаемый тип данных результата
) -> np.ndarray:                       # Возвращает нормализованный массив
    mean = np.array(mean, dtype=image.dtype)  # Преобразование среднего в массив нужного типа
    std = np.array(std, dtype=image.dtype)    # Преобразование std в массив нужного типа
    image = (image - mean) / std       # Нормализация: (x - mean) / std
    return image                       # Возвращает нормализованное изображение


# Функция обработки списка изображений
def process_images(
        images: List[Image.Image],     # Список входных изображений в формате PIL
        size: Dict[str, int] = None,   # Словарь с размерами (height, width)
        resample: Image.Resampling = None,  # Метод интерполяции
        rescale_factor: float = None,  # Коэффициент масштабирования
        image_mean: Optional[Union[float, List[float]]] = None,  # Среднее для нормализации
        image_std: Optional[Union[float, List[float]]] = None,   # Std для нормализации
) -> List[np.ndarray]:                 # Возвращает список обработанных массивов
    height, width = size[0], size[1]   # Извлечение высоты и ширины из словаря
    # Изменение размера всех изображений
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Преобразование изображений в массивы numpy
    images = [np.array(image) for image in images]
    # Масштабирование значений изображений
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Нормализация изображений
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Транспонирование массивов (перемещение каналов в начало: HWC -> CHW)
    images = [image.transpose(2, 0, 1) for image in images]
    return images                      # Возвращает список обработанных изображений


class PaliGemmaProcessor:
    # Определение специального токена для изображений как константа класса
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        # Вызов конструктора родительского класса
        super().__init__()
        # Сохранение количества токенов для изображения
        self.image_seq_length = num_image_tokens
        # Сохранение размера изображения
        self.image_size = image_size
        # Создание словаря с дополнительным специальным токеном изображения
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        # Добавление специального токена в токенизатор
        tokenizer.add_special_tokens(tokens_to_add)
        # Создание токенов для обозначения местоположения (от <loc0000> до <loc1023>)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        # Добавление токенов для сегментации (от <seg000> до <seg127>)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]
        # Добавление всех дополнительных токенов в токенизатор
        tokenizer.add_tokens(EXTRA_TOKENS)
        # Получение ID токена изображения
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # Отключение автоматического добавления токенов начала и конца
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        # Сохранение настроенного токенизатора
        self.tokenizer = tokenizer

    def __call__(self, text: List[str],    # Список текстовых запросов
                 images: List[Image.Image],  # Список изображений
                 padding: str = "longest",   # Тип паддинга для токенизации
                 truncation: bool = True,    # Флаг усечения длинных последовательностей
                 ) -> dict:                  # Возвращает словарь с обработанными данными
        # Проверка: должно быть ровно одно изображение и один запрос
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        # Обработка изображения: изменение размера, масштабирование, нормализация
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,  # Использование бикубической интерполяции
            rescale_factor=1 / 255.0,          # Масштабирование в диапазон [0, 1]
            image_mean=IMAGENET_STANDARD_MEAN, # Среднее ImageNet для нормализации
            image_std=IMAGENET_STANDARD_STD    # Std ImageNet для нормализации
        )
        # Объединение обработанных изображений в один тензор
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)
        # Добавление токенов изображения к каждому запросу
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_propmt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        # Токенизация входных строк
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",  # Возвращать тензоры PyTorch
            padding=padding,      # Применение заданного паддинга
            truncation=truncation # Усечение при необходимости
        )
        # Формирование итогового словаря с пиксельными значениями и токенизированными данными
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data        # Возвращает обработанные данные
