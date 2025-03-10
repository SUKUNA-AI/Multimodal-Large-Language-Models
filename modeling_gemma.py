import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List, Dict, Any
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel


# Класс для хранения кэша ключей и значений в механизме внимания
class KVCache():
    def __init__(self) -> None:
        # Инициализация пустых списков для хранения кэша ключей и значений
        self.key_cache: List[torch.Tensor] = []  # Список тензоров ключей для всех слоёв
        self.value_cache: List[torch.Tensor] = []  # Список тензоров значений для всех слоёв

    def num_items(self) -> int:
        # Метод возвращает количество элементов в кэше (длина последовательности)
        if len(self.key_cache) == 0:
            return 0  # Если кэш пуст, возвращаем 0
        else:
            # Возвращаем размер предпоследнего измерения первого тензора в кэше ключей
            return self.key_cache[0].shape[-2]

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Метод обновляет кэш ключей и значений для заданного слоя
        if len(self.value_cache) <= layer_idx:
            # Если кэш для этого слоя ещё не существует, добавляем новые тензоры
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Если кэш уже существует, конкатенируем новые состояния с существующими по предпоследнему измерению
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Возвращаем обновлённые кэш ключей и значений для текущего слоя
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# Конфигурация модели Gemma (текстовая часть PaliGemma)
class GemmaConfig():
    def __init__(self,
                 vocab_size,  # Размер словаря
                 hidden_size,  # Размер скрытого состояния
                 intermediate_size,  # Размер промежуточного слоя в MLP
                 num_hidden_layers,  # Количество скрытых слоёв
                 num_attention_heads,  # Количество голов внимания
                 num_key_value_heads,  # Количество голов для ключей и значений
                 head_dim=256,  # Размерность одной головы внимания
                 max_position_embeddings=8192,  # Максимальная длина последовательности
                 rms_norm_eps=1e-6,  # Эпсилон для RMS нормализации
                 rope_theta=10000.0,  # Параметр для RoPE (Rotary Position Embedding)
                 attention_bias=False,  # Использовать ли смещение в слоях внимания
                 attention_dropout=0.0,  # Вероятность dropout в механизме внимания
                 pad_token_id=None,  # ID токена заполнения (padding)
                 **kwargs,  # Дополнительные параметры
                 ):
        super().__init__()
        # Инициализация параметров конфигурации
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


# Конфигурация мультимодальной модели PaliGemma (визуальная + текстовая части)
class PaliGemmaConfig():
    def __init__(self,
                 vision_config: None,  # Конфигурация визуальной модели (например, Siglip)
                 text_config: None,  # Конфигурация текстовой модели (Gemma)
                 ignore_index=-100,  # Индекс для игнорирования в функции потерь
                 image_token_index=256000,  # Индекс токена изображения в словаре
                 vocab_size=257152,  # Общий размер словаря
                 projection_dim=2048,  # Размер проекции для объединения модальностей
                 hidden_size=2048,  # Размер скрытого состояния
                 pad_token_id=None,  # ID токена заполнения
                 **kwargs,  # Дополнительные параметры
                 ):
        super().__init__()
        # Инициализация параметров мультимодальной конфигурации
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # Инициализация конфигурации визуальной модели (Siglip)
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        # Инициализация конфигурации текстовой модели (Gemma) с заданным pad_token_id
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Вычисление количества токенов изображения на основе размера изображения и патча
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


# Реализация RMS нормализации для модели Gemma
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Инициализация параметров нормализации
        self.eps = eps  # Малое значение для численной стабильности
        self.weight = nn.Parameter(torch.zeros(dim))  # Обучаемый вес нормализации

    def _norm(self, x):
        # Внутренний метод для выполнения RMS нормализации
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # Нормализация по RMS

    def forward(self, x):
        # Прямой проход: нормализация с учётом веса
        output = self._norm(x.float())  # Приведение к float для точности
        output = output * (1.0 + self.weight.float())  # Применение веса
        return output  # Возвращаем нормализованный тензор


# Реализация MLP (многослойного перцептрона) для модели Gemma
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Линейные слои для проекции входных данных
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Проекция для гейта
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Проекция вверх
        self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Проекция вниз

    def forward(self, x):
        # Прямой проход через MLP с активацией GELU
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")  # Применение гейта с GELU
        up = self.up_proj(x)  # Проекция вверх
        return self.down_proj(gate * up)  # Умножение и проекция вниз


# Функция для повторения ключей и значений для механизма Grouped Query Attention
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape  # Размеры входного тензора
    if n_rep == 1:
        return hidden_states  # Если повторение не требуется, возвращаем исходный тензор
    # Расширение тензора для повторения голов ключей/значений
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Переформатирование в итоговый тензор с увеличенным числом голов
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Реализация ротационных позиционных эмбеддингов (RoPE) для модели Gemma
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # Размерность головы внимания
        self.max_position_embeddings = max_position_embeddings  # Максимальная длина последовательности
        self.base = base  # Базовый параметр для RoPE

        # Вычисление обратных частот для позиционных эмбеддингов
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)  # Регистрация как буфер

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # Прямой проход для вычисления косинусов и синусов RoPE
        self.inv_freq.to(x.device)  # Перенос частот на устройство входного тензора
        # Расширение тензоров для вычисления частот позиций
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "meta" else "cpu"

        # Вычисление углов с отключением автокодирования для точности
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(-1, -2)
            emb = torch.cat((freqs, freqs), dim=-1)  # Объединение для получения полных эмбеддингов
            cos = emb.cos()  # Косинус позиций
            sin = emb.sin()  # Синус позиций

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # Возвращаем в исходном типе данных


# Функция для поворота половины значений тензора в RoPE
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]  # Первая половина тензора
    x2 = x[..., x.shape[-1] // 2:]  # Вторая половина тензора
    return torch.cat((-x2, x1), dim=-1)  # Поворот и объединение


# Функция для применения ротационных позиционных эмбеддингов
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Добавление измерения для совместимости
    sin = sin.unsqueeze(unsqueeze_dim)  # Добавление измерения для совместимости
    q_embed = (q * cos) + (rotate_half(q) * sin)  # Применение RoPE к запросам
    k_embed = (k * cos) + (rotate_half(k) * sin)  # Применение RoPE к ключам
    return q_embed, k_embed  # Возвращаем преобразованные тензоры


# Реализация механизма внимания для модели Gemma
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config  # Конфигурация модели
        self.layer_idx = layer_idx  # Индекс слоя

        # Параметры механизма внимания
        self.attention_dropout = config.attention_dropout  # Dropout для внимания
        self.hidden_size = config.hidden_size  # Размер скрытого состояния
        self.num_heads = config.num_attention_heads  # Количество голов внимания
        self.head_dim = config.head_dim  # Размер одной головы
        self.num_key_value_heads = config.num_key_value_heads  # Количество голов ключей/значений
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # Группы для GQA
        self.max_position_embeddings = config.max_position_embeddings  # Максимальная длина последовательности
        self.rope_theta = config.rope_theta  # Параметр RoPE
        self.is_causal = True  # Флаг причинного внимания (causal)

        assert self.hidden_size % self.num_heads == 0  # Проверка делимости размера состояния на головы

        # Линейные слои для проекции запросов, ключей, значений и выхода
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Инициализация ротационных позиционных эмбеддингов
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self,
                hidden_states: torch.Tensor,  # Входные скрытые состояния
                attention_mask: Optional[torch.Tensor] = None,  # Маска внимания
                position_ids: Optional[torch.LongTensor] = None,  # Позиционные идентификаторы
                kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                **kwargs,
                ) -> tuple[Any, Tensor]:
        bsz, q_len = hidden_states.size()  # Размеры батча и длины последовательности

        # Проекция входных состояний в запросы, ключи и значения
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Переформатирование тензоров для многоголового внимания
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Применение ротационных позиционных эмбеддингов
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Обновление кэша ключей и значений, если он предоставлен
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Повторение ключей и значений для Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Вычисление весов внимания
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert attention_mask is not None  # Проверка наличия маски внимания
        attn_weights = attn_weights + attention_mask  # Применение маски внимания

        # Применение softmax и dropout к весам внимания
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Вычисление выходного состояния внимания
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()  # Перестановка измерений
        attn_output = attn_output.view(bsz, q_len, -1)  # Переформатирование в исходный размер
        attn_output = self.o_proj(attn_output)  # Проекция на выходной слой

        return attn_output, attn_weights  # Возвращаем выход и веса внимания


# Декодерный слой модели Gemma
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size  # Размер скрытого состояния

        # Инициализация механизма внимания и MLP
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        # Слои нормализации перед вниманием и MLP
        self.input_layer_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,  # Входные скрытые состояния
                attention_mask: Optional[torch.Tensor] = None,  # Маска внимания
                position_ids: Optional[torch.LongTensor] = None,  # Позиционные идентификаторы
                kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states  # Сохранение остаточного соединения
        hidden_states = self.input_layer_norm(hidden_states)  # Нормализация перед вниманием
        # Применение механизма внимания
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = residual + hidden_states  # Добавление остаточного соединения
        residual = hidden_states  # Сохранение нового остаточного соединения
        hidden_states = self.post_attention_layernorm(hidden_states)  # Нормализация перед MLP
        hidden_states = self.mlp(hidden_states)  # Применение MLP
        hidden_states = residual + hidden_states  # Добавление остаточного соединения

        return hidden_states  # Возвращаем преобразованные состояния


# Основная модель Gemma (без головы для генерации)
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config  # Конфигурация модели
        self.padding_idx = config.pad_token_id  # ID токена заполнения
        self.vocab_size = config.vocab_size  # Размер словаря

        # Слой эмбеддингов для токенов
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Список декодерных слоёв
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Финальная нормализация
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_inputs_embeddings(self):
        # Метод для получения слоя эмбеддингов
        return self.embed_tokens

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,  # Маска внимания
                position_ids: Optional[torch.LongTensor] = None,  # Позиционные идентификаторы
                inputs_embeds: Optional[torch.FloatTensor] = None,  # Входные эмбеддинги
                kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                ) -> torch.FloatTensor:
        hidden_states = inputs_embeds  # Входные эмбеддинги
        # Нормализация входных эмбеддингов
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Проход через все декодерные слои
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)  # Финальная нормализация

        return hidden_states  # Возвращаем скрытые состояния


# Модель Gemma для причинной генерации текста
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Конфигурация модели
        self.model = GemmaModel(config)  # Основная модель Gemma
        self.vocab_size = config.vocab_size  # Размер словаря
        # Голова для предсказания логитов
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        # Метод для получения слоя эмбеддингов
        return self.model.embed_tokens

    def tie_weights(self):
        # Привязка весов головы к весам эмбеддингов для экономии памяти
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,  # Маска внимания
                position_ids: Optional[torch.LongTensor] = None,  # Позиционные идентификаторы
                inputs_embeds: Optional[torch.FloatTensor] = None,  # Входные эмбеддинги
                kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                ) -> dict[str, Any | None]:
        # Проход через основную модель
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs  # Получение скрытых состояний
        logits = self.lm_head(hidden_states)  # Вычисление логитов
        logits = logits.float()  # Приведение к float32

        # Подготовка возвращаемых данных
        return_data = {
            "logits": logits,  # Логиты для предсказания следующего токена
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache  # Добавление кэша, если он использовался

        return return_data  # Возвращаем словарь с результатами


# Проектор для объединения визуальных и текстовых признаков в PaliGemma
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # Линейный слой для проекции визуальных признаков в пространство проекции
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, x):
        # Прямой проход через проектор
        hidden_states = self.linear(image_features)  # Проекция визуальных признаков
        return hidden_states  # Возвращаем преобразованные состояния


# Полная модель PaliGemma для условной генерации
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config  # Конфигурация модели
        # Визуальная башня (например, Siglip)
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # Проектор для объединения модальностей
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size  # Размер словаря

        # Языковая модель (Gemma)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        # ID токена заполнения (по умолчанию -1, если не задан)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        # Привязка весов языковой модели
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self,
                                             image_features: torch.Tensor,  # Признаки изображений
                                             inputs_embeds: torch.Tensor,  # Входные эмбеддинги текста
                                             inputs_ids: torch.Tensor,  # Входные ID токенов
                                             attention_mask: torch.Tensor,  # Маска внимания
                                             kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                                             ):
        # Объединение текстовых эмбеддингов с признаками изображений
        _, _, embed_dim = image_features.shape  # Размерность признаков изображений
        batch_size, sequence_length = inputs_ids.shape  # Размеры батча и последовательности
        dtype, device = inputs_embeds.dtype, inputs_embeds.device  # Тип данных и устройство

        # Нормализация признаков изображений
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # Создание итогового тензора для объединённых эмбеддингов
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype,
                                      device=inputs_embeds.device)

        # Создание масок для текста, изображений и заполнения
        text_mask = (inputs_ids != self.config.image_token_index) & (inputs_ids != self.pad_token_id)
        image_mask = (inputs_ids == self.config.image_token_index)
        pad_mask = (inputs_ids == self.pad_token_id)

        # Расширение масок для совместимости с размерностью эмбеддингов
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Объединение текстовых и визуальных эмбеддингов с учётом масок
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, scaled_image_features)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Создание причинной маски внимания
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Если кэш пуст, создаём полную причинную маску
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Если используется кэш, создаём маску с учётом длины кэша
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        causal_mask = causal_mask.unsqueeze(1)  # Добавление измерения для совместимости

        # Вычисление позиционных идентификаторов
        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]  # Последние позиции из маски
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)  # Добавление измерения батча
        else:
            # Кумулятивная сумма для всех позиций с заменой 0 на 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        # Возвращаем объединённые эмбеддинги, маску внимания и позиционные идентификаторы
        return final_embedding, causal_mask, position_ids

    def forward(self,
                input_ids: torch.LongTensor = None,  # Входные ID токенов
                pixel_values: torch.FloatTensor = None,  # Значения пикселей изображений
                attention_mask: Optional[torch.Tensor] = None,  # Маска внимания
                kv_cache: Optional[KVCache] = None,  # Кэш ключей и значений
                ) -> Tuple:
        # Проверка, что входные данные не содержат заполнения
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Получение текстовых эмбеддингов
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # Обработка изображений через визуальную башню
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # Проекция визуальных признаков
        image_features = self.multi_modal_projector(selected_image_feature)
        # Объединение текстовых и визуальных данных
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features,
                                                                                                inputs_embeds,
                                                                                                input_ids,
                                                                                                attention_mask,
                                                                                                kv_cache)

        # Проход через языковую модель
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )

        return outputs  # Возвращаем результаты языковой модели

