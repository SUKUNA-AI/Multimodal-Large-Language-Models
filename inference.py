from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


# Функция для перемещения входных данных модели на указанное устройство
def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Перемещает все тензоры из словаря входных данных модели на заданное устройство (CPU/GPU).

    Args:
        model_inputs (dict): Словарь с входными данными модели (например, input_ids, pixel_values).
        device (str): Устройство, на которое нужно переместить данные (например, 'cuda', 'cpu').

    Returns:
        dict: Обновлённый словарь с тензорами, перемещёнными на указанное устройство.
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


# Функция для подготовки входных данных модели
def get_model_inputs(
        processor: 'PaliGemmaProcessor',  # Процессор для подготовки данных (предполагается определённым)
        prompt: str,  # Текстовый запрос
        image_file_path: str,  # Путь к файлу изображения
        device: str  # Устройство для обработки
):
    """
    Подготавливает входные данные для модели PaliGemma из текстового запроса и изображения.

    Args:
        processor: Экземпляр PaliGemmaProcessor для токенизации и обработки данных.
        prompt (str): Текстовый запрос для генерации.
        image_file_path (str): Путь к файлу изображения на диске.
        device (str): Устройство для размещения данных (например, 'cuda', 'cpu').

    Returns:
        dict: Словарь с подготовленными входными данными (input_ids, attention_mask, pixel_values).
    """
    # Открытие изображения из файла
    image = Image.open(image_file_path)
    images = [image]  # Создание списка из одного изображения
    prompts = [prompt]  # Создание списка из одного текстового запроса

    # Обработка текста и изображения с помощью процессора
    model_inputs = processor(text=prompts, images=images)

    # Перемещение данных на указанное устройство
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


# Функция для тестирования генерации текста моделью
def test_inference(
        model: 'PaliGemmaForConditionalGeneration',  # Модель PaliGemma
        processor: 'PaliGemmaProcessor',  # Процессор для данных
        device: str,  # Устройство
        prompt: str,  # Текстовый запрос
        image_file_path: str,  # Путь к изображению
        max_tokens_to_generate: int,  # Максимальное количество генерируемых токенов
        temperature: float,  # Температура для сэмплирования
        top_p: float,  # Параметр top-p для сэмплирования
        do_sample: bool,  # Флаг для выбора между сэмплированием и жадным выбором
):
    """
    Выполняет генерацию текста на основе изображения и текстового запроса.

    Args:
        model: Экземпляр модели PaliGemmaForConditionalGeneration.
        processor: Экземпляр PaliGemmaProcessor для подготовки данных.
        device (str): Устройство для вычислений ('cuda', 'cpu', 'mps').
        prompt (str): Текстовый запрос.
        image_file_path (str): Путь к файлу изображения.
        max_tokens_to_generate (int): Максимальное количество токенов для генерации.
        temperature (float): Температура для управления случайностью сэмплирования.
        top_p (float): Параметр top-p для фильтрации вероятностей.
        do_sample (bool): Если True, используется сэмплирование; иначе — жадный выбор.
    """
    # Получение входных данных для модели
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]  # ID токенов
    attention_mask = model_inputs["attention_mask"]  # Маска внимания
    pixel_values = model_inputs["pixel_values"]  # Значения пикселей изображения

    # Инициализация кэша ключей и значений для ускорения генерации
    kv_cache = KVCache()

    # Определение стоп-токена для завершения генерации
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []  # Список для хранения сгенерированных токенов

    # Генерация токенов до достижения лимита или стоп-токена
    for _ in range(max_tokens_to_generate):
        # Получение выходных данных модели
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]  # Обновление кэша
        next_token_logits = outputs["logits"][:, -1, :]  # Логиты для следующего токена

        # Выбор следующего токена
        if do_sample:
            # Применение температуры и top-p сэмплирования
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Жадный выбор максимального логита
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        assert next_token.size() == (1, 1)  # Проверка размера токена
        next_token = next_token.squeeze(0)  # Удаление измерения батча
        generated_tokens.append(next_token)  # Добавление токена в список

        # Остановка, если сгенерирован стоп-токен
        if next_token.item() == stop_token:
            break

        # Обновление входных данных для следующей итерации
        input_ids = next_token.unsqueeze(-1)  # Новый токен как вход
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1  # Обновление маски внимания
        )

    # Объединение сгенерированных токенов в один тензор
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Декодирование токенов в текст
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Вывод результата
    print(prompt + decoded)


# Вспомогательная функция для top-p сэмплирования
def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Выполняет top-p (nucleus) сэмплирование из распределения вероятностей.

    Args:
        probs (torch.Tensor): Тензор вероятностей размером (batch_size, vocab_size).
        p (float): Порог top-p для фильтрации вероятностей.

    Returns:
        torch.Tensor: Сэмплированный индекс токена.
    """
    # Сортировка вероятностей по убыванию
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Вычисление кумулятивной суммы вероятностей
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Создание маски для исключения токенов, превышающих порог p
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0  # Обнуление вероятностей вне top-p
    # Нормализация оставшихся вероятностей
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Сэмплирование индекса токена из top-p распределения
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Получение исходного индекса токена из отсортированного порядка
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# Основная функция для запуска программы
def main(
        model_path: str = None,  # Путь к файлу модели
        prompt: str = None,  # Текстовый запрос
        image_file_path: str = None,  # Путь к изображению
        max_tokens_to_generate: int = 100,  # Максимальное количество токенов
        temperature: float = 0.8,  # Температура для сэмплирования
        top_p: float = 0.9,  # Параметр top-p
        do_sample: bool = False,  # Флаг сэмплирования
        only_cpu: bool = False,  # Использовать только CPU
):
    """
    Основная точка входа для выполнения вывода модели PaliGemma.

    Args:
        model_path (str): Путь к файлу модели (например, из Hugging Face).
        prompt (str): Текстовый запрос для генерации.
        image_file_path (str): Путь к файлу изображения.
        max_tokens_to_generate (int): Максимальное количество токенов для генерации.
        temperature (float): Температура для управления случайностью.
        top_p (float): Параметр top-p для фильтрации.
        do_sample (bool): Использовать сэмплирование или жадный выбор.
        only_cpu (bool): Если True, использовать только CPU.
    """
    # Установка устройства по умолчанию на CPU
    device = "cpu"

    # Проверка доступности GPU, если не принудительно выбран CPU
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"  # Использование CUDA, если доступно
        elif torch.backends.mps.is_available():
            device = "mps"  # Использование MPS (Apple Silicon), если доступно

    print("Device in use: ", device)

    # Загрузка модели и токенизатора
    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)  # Загрузка модели и токенизатора
    model = model.to(device).eval()  # Перемещение модели на устройство и перевод в режим оценки

    # Получение параметров модели для процессора
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)  # Инициализация процессора

    # Выполнение вывода
    print("Running inference")
    with torch.no_grad():  # Отключение вычисления градиентов для экономии памяти
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


# Точка входа для запуска через командную строку
if __name__ == "__main__":
    fire.Fire(main)  # Запуск функции main с использованием библиотеки fire для CLI
