import os
import requests

API_BASE_URL = "http://hyperturing.stanford.edu:8000"
headers = {"X-API-Key": os.environ["A3_API_KEY"]}

training_config = {
    "architecture_config": {
        "attention_bias": False,
        "head_dim": 64,
        "hidden_size": 1472,
        "intermediate_size": 3904,
        "num_attention_heads": 23,
        "num_hidden_layers": 23,
        "num_key_value_heads": 23,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "vocab_size": 32_000,
    },
    "optimizer_config": {
        "lr_scheduler": {
            "peak_value": 5e-3,
            "final_lr_frac": 0.1,
            "warmup_frac": 0.05,
            "init_value": 0.0,
        },
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "eps_root": 1e-8,
        "grad_clip_norm": 1.0,
    },
    "train_batch_size": 64,
    "val_batch_size": 64,
    "n_evals": 16,
    "total_train_tokens": 19508232192,
    "max_runtime_seconds": 172800.0,
    "model_seed": 0,
}

resp = requests.post(
    f"{API_BASE_URL}/final_submission",
    headers=headers,
    json={"training_config": training_config, "predicted_final_loss": 3.036},
)
resp.raise_for_status()
print(resp.json())