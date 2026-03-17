from __future__ import annotations

LABEL_TO_SYMBOL = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

LABEL_TO_NAME = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
    3: "Fusion beat",
    4: "Unknown / unclassifiable beat",
}

CLASS_SYMBOLS = [LABEL_TO_SYMBOL[index] for index in sorted(LABEL_TO_SYMBOL)]
CLASS_NAMES = [LABEL_TO_NAME[index] for index in sorted(LABEL_TO_NAME)]
NUM_CLASSES = len(CLASS_NAMES)
SAMPLE_LENGTH = 187
