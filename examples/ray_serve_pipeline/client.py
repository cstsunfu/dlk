import asyncio
import json

import aiohttp
import requests


def test_single_model(model_name, sentence):
    """Test a single model endpoint."""
    url = f"http://localhost:8000/predict/{model_name}"
    payload = {"sentence": sentence, "uuid": "req-001"}

    response = requests.post(url, json=payload)
    print(f"\n--- Result from {model_name} ---")
    print(json.dumps(response.json(), indent=2))


def test_ensemble(models, sentence):
    """Test the ensemble endpoint."""
    url = "http://localhost:8000/ensemble"
    payload = {
        "models": models,
        "data": {"sentence": sentence, "uuid": "req-ensemble-001"},
    }

    response = requests.post(url, json=payload)
    print(f"\n--- Ensemble Result ---")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    test_sentence = "This is definitely a positive sentence!"

    # 1. Test PyTorch Backend
    test_single_model("bert_pytorch", test_sentence)

    # 2. Test ONNX Backend
    test_single_model("bert_onnx", test_sentence)

    # 3. Test Both (Ensemble Request)
    test_ensemble(["bert_pytorch", "bert_onnx"], test_sentence)
