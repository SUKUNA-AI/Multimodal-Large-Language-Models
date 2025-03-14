from typing import Optional, Tuple
import torch
import torch.nn as nn


#%%
class SiglipVisionConfig:
    def __init__(self,
                 hidden_size=768,  # Размер скрытого слоя —  задаёт размер эмбеддингов
                 intermediate_size=3072,  # Размер промежуточного слоя
                 num_hidden_layers=12,  # Количество скрытых слоёв, как уровни наших эмоций, для глубокой обработки
                 num_attention_heads=12,  # Число "голов" внимания
                 num_channels=3,  # Количество цветовых каналов (RGB)
                 image_size=224,  # Размер изображения
                 patch_size=16,  # Размер патча, то есть размер кусочка изображения
                 layer_norm_eps=1e-6,  # Мелкое значение для стабильности нормализации
                 attention_dropout=0.0,  # Дропаут в механизме внимания — для защиты от переобучения
                 num_image_tokens: int = None,  # Количество токенов изображения
                 **kwargs  # Дополнительные аргументы для гибкости и будущих фич
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # Сохраняем конфигурацию модели
        self.embed_dim = config.hidden_size  # Размерность эмбеддингов (скрытый размер)
        self.image_size = config.image_size  # Размер входного изображения (например, 224x224)
        self.patch_size = config.patch_size  # Размер патча (например, 16x16)

        # Слой для преобразования патчей изображения в эмбеддинги
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # Количество входных каналов (например, 3 для RGB)
            out_channels=self.embed_dim,  # Количество выходных каналов (размерность эмбеддинга)
            kernel_size=self.patch_size,  # Размер ядра свёртки (равен размеру патча)
            stride=self.patch_size,  # Шаг свёртки (равен размеру патча, чтобы патчи не перекрывались)
            padding="valid"  # Без дополнения (padding), выходной размер уменьшается
        )

        # Вычисляем количество патчей: (размер изображения / размер патча)²
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches  # Количество позиций равно количеству патчей

        # Слой для позиционных эмбеддингов, чтобы сохранить информацию о положении патчей
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # Регистрируем буфер для идентификаторов позиций (position_ids), используемых для индексации позиционных эмбеддингов
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),  # Форма: [1, num_positions]
            persistent=False  # Не сохраняется в state_dict модели
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Вход: pixel_values с формой [Batch_size, Channels, Height, Width]
        _, _, height, width = pixel_values.shape  # Извлекаем размеры входного тензора

        # Применяем слой patch_embedding для преобразования изображения в эмбеддинги патчей
        # Выход: [Batch_size, embed_dim, H', W'], где H' = height // patch_size, W' = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)

        # Преобразуем пространственные измерения в плоский вид и транспонируем
        # [Batch_size, embed_dim, H', W'] -> [Batch_size, num_patches, embed_dim]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Добавляем позиционные эмбеддинги к эмбеддингам патчей
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # Возвращаем итоговые эмбеддинги с формой [Batch_size, num_patches, embed_dim]
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size  # Размерность эмбеддингов (скрытый размер)
        self.num_heads = config.num_attention_heads  # Количество голов внимания
        self.head_dim = self.embed_dim // self.num_heads  # Размерность на одну голову внимания
        self.scale = self.head_dim ** -0.5  # Масштабирующий коэффициент для внимания (1/sqrt(head_dim))
        self.dropout = config.attention_dropout  # Вероятность выброса (dropout) для весов внимания

        # Линейные слои для проекции скрытых состояний в запрос (query), ключ (key) и значение (value)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Проекция для ключей
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Проекция для значений
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Проекция для запросов
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Выходная проекция после внимания

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Вход: hidden_states с формой [Batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()  # Извлекаем размеры входного тензора

        # Проецируем скрытые состояния в запросы, ключи и значения
        query_states = self.q_proj(hidden_states)  # [Batch_size, seq_len, embed_dim]
        key_states = self.k_proj(hidden_states)  # [Batch_size, seq_len, embed_dim]
        value_states = self.v_proj(hidden_states)  # [Batch_size, seq_len, embed_dim]

        # Преобразуем для многоголового внимания: [Batch_size, seq_len, num_heads, head_dim]
        # Затем транспонируем в [Batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Вычисляем веса внимания: [Batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Проверяем правильность формы весов внимания
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Веса внимания должны иметь размер {(batch_size, self.num_heads, seq_len, seq_len)}, "
                f"но имеют размер {attn_weights.size()}"
            )

        # Применяем softmax для получения вероятностей внимания
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Применяем dropout к весам внимания
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Вычисляем выход внимания: [Batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Транспонируем обратно в [Batch_size, seq_len, num_heads, head_dim] и делаем память непрерывной
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Преобразуем в исходную форму: [Batch_size, seq_len, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Применяем выходную проекцию
        attn_output = self.out_proj(attn_output)

        # Возвращаем выход внимания и веса внимания
        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Сохраняем конфигурацию модели

        # Первый линейный слой: проекция из hidden_size в intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)

        # Второй линейный слой: проекция обратно из intermediate_size в hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Вход: hidden_states с формой [Batch_size, seq_len, hidden_size]

        # Применяем первый линейный слой
        hidden_states = self.fc1(hidden_states)

        # Применяем активацию GELU с приближением через tanh
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        # Применяем второй линейный слой
        hidden_states = self.fc2(hidden_states)

        # Возвращаем результат с формой [Batch_size, seq_len, hidden_size]
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # Размерность эмбеддингов (скрытый размер)

        # Механизм самовнимания
        self.self_attn = SiglipAttention(config)

        # Нормализация слоя перед вниманием
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # Многослойный перцептрон (MLP)
        self.mlp = SiglipMLP(config)

        # Нормализация слоя перед MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Вход: hidden_states с формой [Batch_size, seq_len, embed_dim]

        # Сохраняем остаточную связь (residual) для последующего добавления
        residual = hidden_states

        # Применяем нормализацию слоя
        hidden_states = self.layer_norm1(hidden_states)

        # Применяем механизм самовнимания
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # Добавляем остаточную связь
        hidden_states = residual + hidden_states

        # Сохраняем новую остаточную связь
        residual = hidden_states

        # Применяем нормализацию слоя перед MLP
        hidden_states = self.layer_norm2(hidden_states)

        # Применяем MLP
        hidden_states = self.mlp(hidden_states)

        # Добавляем остаточную связь
        hidden_states = residual + hidden_states

        # Возвращаем результат с формой [Batch_size, seq_len, embed_dim]
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # Сохраняем конфигурацию модели

        # Создаём список слоёв энкодера
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Вход: inputs_embeds с формой [Batch_size, seq_len, embed_dim]

        hidden_states = inputs_embeds  # Инициализируем скрытые состояния входными эмбеддингами

        # Проходим через каждый слой энкодера
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        # Возвращаем итоговые скрытые состояния с формой [Batch_size, seq_len, embed_dim]
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # Сохраняем конфигурацию модели
        embed_dim = config.hidden_size  # Размерность эмбеддингов

        # Слой для преобразования изображения в эмбеддинги патчей
        self.embeddings = SiglipVisionEmbeddings(config)

        # Энкодер, состоящий из нескольких слоёв
        self.encoder = SiglipEncoder(config)

        # Пост-нормализация слоя после энкодера
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Вход: pixel_values с формой [Batch_size, Channels, Height, Width]

        # Преобразуем пиксельные значения в эмбеддинги патчей
        # Выход: [Batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        # Пропускаем через энкодер
        # Выход: [Batch_size, num_patches, embed_dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Применяем пост-нормализацию слоя
        # Выход: [Batch_size, num_patches, embed_dim]
        last_hidden_state = self.post_layer_norm(last_hidden_state)

        # Возвращаем итоговые скрытые состояния
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # Сохраняем конфигурацию модели

        # Модель визуального трансформера
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # Вход: pixel_values с формой [Batch_size, Channels, Height, Width]

        # Пропускаем через визуальный трансформер
        # Выход: [Batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)
