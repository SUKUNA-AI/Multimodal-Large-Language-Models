import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, KVCache
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler


# Кастомный датасет для пар "изображение — текст"
class ImageTextDataset(Dataset):
    def __init__(self, data_dir, image_files, captions, processor):
        self.data_dir = data_dir
        self.image_files = image_files  # Список путей к изображениям
        self.captions = captions  # Список подписей
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(text=[caption], images=[image])
        return {
            "input_ids": inputs["input_ids"][0],
            "pixel_values": inputs["pixel_values"][0],
            "attention_mask": inputs["attention_mask"][0]
        }


# Функция обучения
def train_model(model, processor, dataloader, device, num_epochs=1, lr=2e-5, accumulation_steps=4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()  # Для mixed precision
    loss_fn = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"].half()  # FP16 для экономии памяти
            attention_mask = inputs["attention_mask"]

            with autocast():  # Mixed precision
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask
                )
                logits = outputs["logits"]

                # Сдвиг меток для предсказания следующего токена
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = processor.tokenizer.pad_token_id

                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / accumulation_steps  # Нормализация для накопления градиентов

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
        torch.save(model.state_dict(), f"paligemma_epoch_{epoch + 1}.pth")

    return model


# Основная функция
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "$HOME/projects/paligemma-weights/paligemma-3b-pt-224"

    # Загрузка модели и токенизатора
    model, tokenizer = load_hf_model(model_path, device)
    processor = PaliGemmaProcessor(tokenizer, model.config.vision_config.num_image_tokens,
                                   model.config.vision_config.image_size)

    # Оптимизация под T4
    model = model.to(device).half()  # FP16
    model.gradient_checkpointing_enable()  # Gradient checkpointing

    # Пример данных (замени на свой датасет)
    data_dir = "path/to/coco/train2017"
    image_files = [f"{data_dir}/image1.jpg", f"{data_dir}/image2.jpg"]  # Список путей
    captions = ["A tall building", "A sunny park"]  # Список подписей
    dataset = ImageTextDataset(data_dir, image_files, captions, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 для T4

    # Обучение
    model = train_model(model, processor, dataloader, device, num_epochs=1)


if __name__ == "__main__":
    main()
