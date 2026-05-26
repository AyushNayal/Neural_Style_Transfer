import torch

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Image ────────────────────────────────────────────────────────────────────
MAX_SIZE = 512          # images larger than this get resized (preserving aspect ratio)

# ─── Optimization ─────────────────────────────────────────────────────────────
NUM_STEPS      = 2000   # number of Adam iterations
LEARNING_RATE  = 0.005  # Adam learning rate

# ─── Loss Weights ─────────────────────────────────────────────────────────────
STYLE_WEIGHT   = 1e6    # how strongly to enforce style
CONTENT_WEIGHT = 2.0    # how strongly to preserve content structure
TV_WEIGHT      = 1e-5   # smoothness penalty (total variation)

# ─── Per-layer Style Weights ──────────────────────────────────────────────────
# Higher weight on deeper layers = more abstract/global style
# Higher weight on shallow layers = more texture/color style
STYLE_WEIGHTS = {
    'conv_1': 0.1,   # low-level: colors, fine textures
    'conv_2': 1.5,   # mid-level: texture patterns
    'conv_3': 1.5,   # mid-level: larger patterns
    'conv_4': 3.7,   # high-level: style structure
    'conv_5': 4.0,   # highest-level: global style
}

# ─── Output ───────────────────────────────────────────────────────────────────
SAVE_EVERY     = 500    # save intermediate image every N steps
OUTPUT_DIR     = "outputs"