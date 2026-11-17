"""Canonical model names and question templates used across pipeline stages."""
MODEL_NAMES = {'lfm': 'LFM2_5_1_2B', 'qwen': 'QWEN3_5_9B', 'gemma': 'GEMMA4_31B', 'ministral': 'MINISTRAL3_3B'}
QUESTIONS = {'birth': 'When was {person} born?', 'death': 'When did {person} die?',
             'birthplace': 'Where was {person} born?', 'deathplace': 'Where did {person} die?'}
GRANOLA_TRAIN_RELATIONS = {'P112', 'P170', 'P175', 'P176', 'P40', 'P69'}
MODEL_IDS = {
    'LFM2_5_1_2B': 'LiquidAI/LFM2.5-1.2B-Instruct',
    'GEMMA4_31B': 'google/gemma-4-31B-it',
    'QWEN3_5_9B': 'Qwen/Qwen3.5-9B',
    'MINISTRAL3_3B': 'mistralai/Ministral-3-3B-Instruct-2512-BF16',
}
