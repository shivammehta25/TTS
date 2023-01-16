from pathlib import Path

import torch
from tqdm.auto import tqdm

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

MODELS = {
    # 'glow': "tts_models/en/ljspeech/glow-tts",
    # 'tacotron2': "tts_models/en/ljspeech/tacotron2-DCA",
    # 'vits': "tts_models/en/ljspeech/vits",
    "overflow": "tts_models/en/ljspeech/overflow",
    # 'fast_pitch': "tts_models/en/ljspeech/fast_pitch",
}

MODELS_PATH = {
    "overflow": {
        "model_path": "recipes/ljspeech/overflow/overflow_ljspeech-December-10-2022_09+42AM-c2df9f39/best_model_279889.pth",
        "config_path": "recipes/ljspeech/overflow/overflow_ljspeech-December-10-2022_09+42AM-c2df9f39/config.json",
    }
}

VOCODER = "vocoder_models/en/ljspeech/hifigan_v2"
FOLDER = Path("synth_output")
FOLDER.mkdir(exist_ok=True, parents=True)


test_sentences = {
    "1": "This is a test sentence.",
    "2": "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
}


manager = ModelManager()
pbar = tqdm(MODELS.items())
for model_name, model in pbar:
    pbar.set_description(f"{model_name}")
    if model_name in MODELS_PATH:
        model_path = MODELS_PATH[model_name]["model_path"]
        config_path = MODELS_PATH[model_name]["config_path"]
        model_item = {}
        model_item["default_vocoder"] = VOCODER
    else:
        model_path, config_path, model_item = manager.download_model(model)

    vocoder_name = model_item["default_vocoder"]
    if vocoder_name is not None:
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    synthesiser = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=vocoder_path if vocoder_name is not None else None,
        vocoder_config=vocoder_config_path if vocoder_name is not None else None,
        use_cuda=True,
    )
    for idx, text in test_sentences.items():
        wav = synthesiser.tts(text, split=False)
        save_folder = FOLDER / model_name
        save_folder.mkdir(exist_ok=True, parents=True)
        synthesiser.save_wav(wav, save_folder / f"{idx}.wav")
