# Multimodal Large Language Models 🚀

![PaliGemma](https://img.shields.io/badge/Model-PaliGemma-blue) 
![License](https://img.shields.io/badge/License-MIT-green) 
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Multimodal](https://img.shields.io/badge/Multimodal-Vision+Text-orange)

Привет! Добро пожаловать в **Multimodal Large Language Models** — проект, посвящённый использованию мультимодальной модели [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) от Google. Здесь ты найдёшь всё, чтобы запустить готовую модель с Hugging Face, а также полное описание её архитектуры для любопытных умов! 😎 Мы не будем обучать модель с нуля (зачем, если есть готовая?), но дадим тебе возможность поиграться с инференсом и понять, как это работает под капотом.

---

## О проекте 🌟

[PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) — это компактная мультимодальная модель (3 миллиарда параметров), которая объединяет визуальные и текстовые данные. Она построена на базе Vision Transformer (SigLIP) для обработки изображений и языковой модели Gemma для генерации текста. Этот проект вдохновлён видео [Coding Multimodal AI From Scratch](https://www.youtube.com/watch?v=vAmKB7iPkWw), но вместо создания модели с нуля мы используем готовую реализацию от Google через Hugging Face.

### Что здесь есть?
- **Инференс**: Простой скрипт для генерации текста по изображениям и запросам.
- **Исходный код**: Полная реализация PaliGemma (визуальная часть SigLIP + текстовая часть Gemma) для тех, кто хочет копнуть глубже.
- **Подробное описание**: Как работает архитектура PaliGemma — от входных данных до выхода. Читай ниже! 📚

---

## ⚡ Быстрый старт

### Установка
```bash
git clone https://github.com/SUKUNA-AI/Multimodal-Large-Language-Models.git
cd Multimodal-Large-Language-Models
pip install -r requirements.txt
```

### Запуск примера
```bash
./launch_inference.sh
```
### Или через Python:
```python
from inference import main

main(
    model_path="paligemma-3b-pt-224",
    prompt="What is happening in this image?",
    image_file_path="test_images/astronaut.jpg",
    max_tokens_to_generate=100
)
```
## Использование 🎉

### Пример с дефолтными параметрами(см.inference.py):
```python
    --model_path "paligemma-3b-pt-224" \
    --prompt "What is in this image?" \
    --image_file_path "test_images/cat.jpg" # Или URL: "https://example.com/image.jpg"
 ```   
### Пример вывода:
#### **A cat sitting on a windowsill with a city view in the background.**

### Или используй Bash-скрипт:
```bash
chmod +x run_inference.sh
./run_inference.sh
```
---

# Как работает PaliGemma? Полное описание архитектуры 🧠

PaliGemma принимает изображение и текстовый запрос, а выдаёт текст (например, описание изображения или ответ на вопрос). Давай разберём её по шагам! 😊

---

## 1. Входные данные 🚀

- **Изображение**: Картинка (например, 224x224 пикселя) в формате RGB. 🎨
- **Текст**: Запрос (например, "Describe this image"), токенизированный в ID. 🔤

---

## 2. Визуальная часть: SigLIP (Vision Transformer) 👁️

**SigLIP преобразует изображение в набор эмбеддингов:**

- **Разбиение на патчи**: Изображение делится на 196 патчей (14x14, каждый 16x16 пикселей). 📸
- **Эмбеддинги патчей**: Каждый патч превращается в вектор размером 1152 через свёрточный слой (`nn.Conv2d`). 🧮
- **Позиционные эмбеддинги**: Добавляются с помощью `nn.Embedding`. 📍
- **Трансформер**: 12 слоёв с механизмом самовнимания (self-attention), включая:
  - **Multi-Head Attention** 🔀
  - **MLP с нелинейностью** 💥
  - **LayerNorm для стабильности** ⚖️
- **Выход**: 196 векторов по 1152 элементов — визуальные признаки. ✨

![Vision Transformer Diagram](https://github.com/user-attachments/assets/2410285d-d5e5-4dcb-85af-08659184d581)

---

## 3. Проектор: MultiModalProjector 🌉

- **Функция**: Визуальные признаки (1152) проецируются в пространство размером 2048 через `nn.Linear`.
- **Назначение**: Обеспечивает связь между визуальными и текстовыми данными. 🤝


---

## 4. Текстовая часть: Gemma (Language Model) 📝

### Gemma генерирует текст:

- **Токенизация**: Запрос преобразуется в ID токенов. 🔠
- **Эмбеддинги**: Токены становятся векторами размером 2048 через `nn.Embedding`. 🔢
- **Объединение**: Токены `<image>` заменяются визуальными признаками. 🔄
- **Трансформер**: 18 слоёв с:
  - **Rotary Positional Embeddings (RoPE)** 🎡
  - **Grouped Query Attention (GQA)** 🤹
  - **MLP с GELU** ⚡
  - **RMSNorm** 🛡️
  - **Голова (LM Head)**: Преобразует скрытые состояния в логиты (словарь: 257152 токена). 📊

![Gemma Language Model](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/image6_iinFugc.original.png)

---

## 5. Генерация текста ✍️

- Модель предсказывает токены с помощью **жадного выбора** или **сэмплирования (top-p/top-k)**. 🎯
- Генерация останавливается по достижении `max_tokens` или при встрече стоп-токена (EOS). 🏁

![Text Generation](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/image3_3kHryqa.original.png)

---

## 6. Кэш KV (KVCache) ⚡

- **Функция**: Хранит ключи и значения внимания для ускорения авторегрессионной генерации, обеспечивая эффективную работу модели. 🚀


---

## Итог 🔥

- **Вход**: Изображение + "Describe this image".
- **Выход**: "A modern building with large windows."
- **Сложность**: SigLIP (1B) + Проектор + Gemma (2B) = ~3B параметров.

Погрузись в этот мир нейросетевых чудес, и пусть каждая деталь архитектуры вдохновляет тебя на новые свершения! 🌟😻

---

## Почему использовать готовую модель? 🤔
- **Экономия времени: Обучение требует огромных ресурсов.**
- **Качество: PaliGemma уже обучена на гигантских датасетах.**
- **Простота: Hugging Face делает запуск лёгким.**
  
---

## Дополнительные файлы 🛠️
- [modeling_gemma.py](https://github.com/SUKUNA-AI/Multimodal-Large-Language-Models/blob/main/modeling_gemma.py): Полная реализация PaliGemma.
- [processing_paligemma.py](https://github.com/SUKUNA-AI/Multimodal-Large-Language-Models/blob/main/processing_paligemma.py): Кастомный процессор данных.
- [siglip.py](https://github.com/SUKUNA-AI/Multimodal-Large-Language-Models/blob/main/siglip.py): Реализация SigLIP.
- [utils.py](https://github.com/SUKUNA-AI/Multimodal-Large-Language-Models/blob/main/utils.py): Утилита для загрузки весов (SafeTensors).
- Эти файлы необязательны для базового инференса, но полезны для изучения и кастомизации! 😉

---

## Вдохновение ✨
- Проект вдохновлён видео [Coding Multimodal AI From Scratch](https://www.youtube.com/watch?v=vAmKB7iPkWw), где автор показывает, как строить мультимодальные модели. Мы упростили задачу, взяв готовую PaliGemma, но сохранили дух исследования!
---
## Лицензия 📜
Проект распространяется под лицензией MIT.
---
## Благодарности 🙏
- [Google](https://huggingface.co/google) за PaliGemma.
- [Hugging Face](https://huggingface.co) за удобный доступ к моделям.
- Автору [Coding Multimodal AI From Scratch](https://www.youtube.com/watch?v=vAmKB7iPkWw) за вдохновение.
---

© 2025 SUKUNA-AI — с любовью к технологиям! ❤️





