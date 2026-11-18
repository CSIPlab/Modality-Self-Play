"""Qwen3 special token ids used for modality injection.

Single source for the ids shared by the datasets, the prompt builders and
the model wrappers.
"""

IM_START = 151644
IM_END = 151645
PAD = 151643

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655   # vision feature position
FIM_PAD = 151662     # repurposed as the audio feature position

MODALITY_TOKEN_IDS = {
    "vision": IMAGE_PAD,
    "audio": FIM_PAD,
}

MODALITY_TOKEN_STRINGS = {
    "vision": "<|image_pad|>",
    "audio": "<|fim_pad|>",
}
