# %% CCT parameters

projection_dim = 64
transformer_units = [projection_dim, projection_dim]

# %% Training parameters

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
INPUT_SHAPE = (128, 431, 1)
warmup_p = 0.15
lr_start = 1e-5
lr_max = 1e-3
weight_decay = 2e-4
test_size_split = 0.2

# %% Dictioonary of classes

DICT_LABELS = {
    "classical": 0, 
    "flamenco": 1, 
    "hiphop": 2, 
    "jazz": 3, 
    "pop": 4,
    "r&b": 5,
    "reggaeton": 6
}

DICT_LABELS_PREDICT = {
    0: "classical", 
    1: "flamenco", 
    2: "hiphop", 
    3: "jazz", 
    4: "pop",
    5: "r&b",
    6: "reggaeton"
}