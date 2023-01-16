import argparse
import importlib
import os
from argparse import RawTextHelpFormatter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from TTS.config import BaseAudioConfig, BaseDatasetConfig, load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.models import setup_model
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_checkpoint

if __name__ == "__main__":
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Extract attention masks from trained Tacotron/Tacotron2 models.
These masks can be used for different purposes including training a TTS model with a Duration Predictor.\n\n"""
        """Each attention mask is written to the same path as the input wav file with ".npy" file extension.
(e.g. path/bla.wav (wav file) --> path/bla.npy (attention mask))\n"""
        """
Example run:
    CUDA_VISIBLE_DEVICE="0" python TTS/bin/compute_attention_masks.py
        --model_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/checkpoint_200000.pth
        --config_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/config.json
        --dataset_metafile metadata.csv
        --data_path /root/LJSpeech-1.1/
        --batch_size 32
        --dataset ljspeech
        --use_cuda True
""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to Tacotron/Tacotron2 model file ")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Tacotron/Tacotron2 config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        required=True,
        help="Target dataset processor name from TTS.tts.dataset.preprocess.",
    )

    parser.add_argument(
        "--dataset_metafile",
        type=str,
        default="",
        required=True,
        help="Dataset metafile inclusing file paths with transcripts.",
    )
    parser.add_argument("--data_path", type=str, default="", help="Defines the data path. It overwrites config.json.")
    parser.add_argument("--use_cuda", type=bool, default=False, help="enable/disable cuda.")

    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for the model. Use batch_size=1 if you have no CUDA."
    )
    args = parser.parse_args()

    C = load_config(args.config_path)
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    print(C.keys())
    # import pdb; pdb.set_trace()
    if "characters" in C.keys():
        # symbols, phonemes = make_symbols(**C.characters)
        symbols, phonemes = C.characters.characters, C.characters.phonemes

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    # TODO: handle multi-speaker
    model = setup_model(C)
    model, _ = load_checkpoint(model, args.model_path, args.use_cuda, True)

    # # data loader
    # preprocessor = importlib.import_module("TTS.tts.datasets.formatters")
    # preprocessor = getattr(preprocessor, args.dataset)
    # meta_data = preprocessor(args.data_path, args.dataset_metafile)
    C.phoneme_cache_path = "TTS/bin/phoneme_cache"
    
    tokenizer, config = TTSTokenizer.init_from_config(C)
    
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join('data', "LJSpeech-1.1")
    )
   
    
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=False,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model.tokenizer = tokenizer
    loader = model.get_data_loader(config, {}, True, train_samples, True, 1)

    # import pdb; pdb.set_trace()
    # dataset = TTSDataset(
    #     model.decoder.r,
    #     C.text_cleaner,
    #     compute_linear_spec=False,
    #     ap=ap,
    #     meta_data=meta_data,
    #     characters=C.characters if "characters" in C.keys() else None,
    #     add_blank=C["add_blank"] if "add_blank" in C.keys() else False,
    #     use_phonemes=C.use_phonemes,
    #     phoneme_cache_path=C.phoneme_cache_path,
    #     phoneme_language=C.phoneme_language,
    #     enable_eos_bos=C.enable_eos_bos_chars,
    # )
    # compute attentions
    file_paths = []
    with torch.no_grad():
        for data in tqdm(loader):
            # setup input data
            text_input = data['token_id']
            text_lengths = data['token_id_lengths']
            linear_input = data['linear']
            mel_input = data['mel']
            mel_lengths = data['mel_lengths']
            stop_targets = data['stop_targets']
            item_idxs = data['item_idxs']

            # dispatch data to GPU
            if args.use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            model_outputs = model.forward(text_input, text_lengths, mel_input)

            alignments = model_outputs["alignments"].detach()
            for idx, alignment in enumerate(alignments):
                item_idx = item_idxs[idx]
                # interpolate if r > 1
                alignment = (
                    torch.nn.functional.interpolate(
                        alignment.transpose(0, 1).unsqueeze(0),
                        size=None,
                        scale_factor=model.decoder.r,
                        mode="nearest",
                        align_corners=None,
                        recompute_scale_factor=None,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                # remove paddings
                alignment = alignment[: mel_lengths[idx], : text_lengths[idx]].cpu().numpy()
                # set file paths
                wav_file_name = os.path.basename(item_idx)
                align_file_name = os.path.splitext(wav_file_name)[0] + "_attn.npy"
                file_path = item_idx.replace(wav_file_name, align_file_name)
                # save output
                wav_file_abs_path = os.path.abspath(item_idx)
                file_abs_path = os.path.abspath(file_path)
                file_paths.append([wav_file_abs_path, file_abs_path])
                np.save(file_path, alignment)

        # ourput metafile
        metafile = os.path.join(args.data_path, "metadata_attn_mask.txt")

        with open(metafile, "w", encoding="utf-8") as f:
            for p in file_paths:
                f.write(f"{p[0]}|{p[1]}\n")
        print(f" >> Metafile created: {metafile}")
