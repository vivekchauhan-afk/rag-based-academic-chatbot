"""
=========================================================
Callbacks
=========================================================
"""

from transformers import EarlyStoppingCallback


def get_callbacks():

    callbacks = [

        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.0
        )

    ]

    return callbacks