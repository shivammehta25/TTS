import torch

from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.io import load_checkpoint
from TTS.vocoder.models import setup_model as setup_vocoder

sentences = {
    "LJ034-0035.wav": "The position of this palmprint on the carton was parallel with the long axis of the box and at right angles with the short axis.",
    "LJ023-0033.wav": "We will not allow ourselves to run around in new circles of futile discussion and debate, always postponing the day of decision.",
    "LJ046-0055.wav": "It is now possible for Presidents to travel the length and breadth of a land far larger than the United States.",  #! new
    "LJ016-0277.wav": "This is proved by contemporary accounts, especially one graphic and realistic article which appeared in the 'Times'.",
    "LJ050-0022.wav": "A formal and thorough description of the responsibilities of the advance agent is now in preparation by the Service.",
    "LJ002-0225.wav": "The rentals of rooms and fees went to the warden, whose income was two thousand three hundred seventy two pounds.",
    "LJ021-0012.wav": "With respect to industry and business, but nearly all are agreed that private enterprise in times such as these.",  #! new
    "LJ035-0155.wav": "The only exit from the office in the direction Oswald was moving was through the door to the front stairway.",  #! new
    "LJ028-0421.wav": "it was the beginning of the great collections of Babylonian antiquities in the museums of the Western world.",
    "LJ015-0025.wav": "The bank enjoyed an excellent reputation, it had a good connection, and was supposed to be perfectly sound.",
    "LJ016-0054.wav": "But he did not like the risk of entering a room by the fireplace, and the chances of detection it offered.",
    "LJ006-0021.wav": "Later on, he had devoted himself to the personal investigation of the prisons of the United States.",
    "LJ016-0274.wav": "And the windows of the opposite houses, which commanded a good view, as usual fetched high prices.",
    "LJ002-0171.wav": "The boy declared he saw no one, and accordingly passed through without paying the toll of a penny.",
    "LJ021-0139.wav": "There should be at least a full and fair trial given to these means of ending industrial warfare.",
    "LJ003-0282.wav": "Many years were to elapse before these objections should be fairly met and universally overcome.",
    "LJ016-0367.wav": "Under the new system the whole of the arrangements from first to last fell upon the officers.",  #! new
    "LJ021-0025.wav": "And in many directions, the intervention of that organized control which we call government",
    "LJ021-0140.wav": "and in such an effort we should be able to secure for employers and employees and consumers.",
    "LJ010-0027.wav": "Nor did the methods by which they were perpetrated greatly vary from those in times past.",
    "LJ029-0004.wav": "The narrative of these events is based largely on the recollections of the participants.",
    "LJ046-0191.wav": "It had established periodic regular review of the status of four hundred individuals.",
    "LJ030-0063.wav": "He had repeated this wish only a few days before, during his visit to Tampa, Florida.",
    "LJ018-0349.wav": "His disclaimer, distinct and detailed on every point, was intended simply for effect.",
    "LJ033-0072.wav": "I then stepped off of it and the officer picked it up in the middle and it bent so.",
    "LJ050-0029.wav": "That is reflected in definite and comprehensive operating procedures.",  #! new
    "LJ021-0040.wav": "The second step we have taken in the restoration of normal business enterprise.",
    "LJ027-0006.wav": "In all these lines the facts are drawn together by a strong thread of unity.",
    "LJ002-0260.wav": "Yet the public opinion of the whole body seems to have checked dissipation.",
    "LJ013-0055.wav": "The jury did not believe him, and the verdict was for the defendants.",
    "LJ008-0215.wav": "One by one the huge uprights of black timber were fitted together.",
    "LJ018-0206.wav": "He was a tall, slender man, with a long face and iron gray hair.",
    "LJ004-0239.wav": "He had been committed for an offense for which he was acquitted.",
    "LJ037-0007.wav": "Three others subsequently identified Oswald from a photograph.",
    "LJ037-0248.wav": "The eyewitnesses vary in their identification of the jacket.",
    "LJ019-0055.wav": "High authorities were in favor of continuous separation.",
    "LJ038-0035.wav": "Oswald rose from his seat, bringing up both hands.",
    "LJ010-0297.wav": "But there were other notorious cases of forgery.",
    "LJ019-0371.wav": "Yet the law was seldom if ever enforced.",
    "LJ018-0159.wav": "This was all the police wanted to know.",
}


def load_model_components(model_path, config_path):
    config = load_config(config_path)
    model = setup_model(config)
    model.load_checkpoint(config, model_path, eval=True)
    model = model.cuda()
    print("Model loaded")
    return model


def load_vocoder(vocoder_path, vocoder_config):
    vocoder_config = load_config(vocoder_config)
    vocoder_model = setup_vocoder(vocoder_config)
    vocoder_model.load_checkpoint(vocoder_config, vocoder_path, eval=True)
    vocoder_model = vocoder_model.cuda()
    print("Vocoder loaded")
    return vocoder_model


def synthesise_test(model, vocoder, text, device="cuda"):
    txt_id = model.tokenizer.text_to_ids(text)
    txt_id = torch.tensor(txt_id, device=device).unsqueeze(0)
    x_len = torch.tensor([txt_id.shape[1]], device=device)
    output = model.inference(txt_id, aux_input={"x_lengths": x_len})
    return output


if __name__ == "__main__":
    # model_path = "recipes/ljspeech/overflow/overflow_ljspeech-December-10-2022_09+42AM-c2df9f39/best_model_279889.pth"
    # config_path = "recipes/ljspeech/overflow/overflow_ljspeech-December-10-2022_09+42AM-c2df9f39/config.json"

    # vocoder_path = "/home/smehta/.local/share/tts/vocoder_models--en--ljspeech--hifigan_v2/model_file.pth"
    # vocoder_config = "/home/smehta/.local/share/tts/vocoder_models--en--ljspeech--hifigan_v2/config.json"

    model_path = "recipes/ljspeech/glow_tts/run-January-13-2023_09+07PM-39a668ff/checkpoint_100000.pth"
    config_path = "recipes/ljspeech/glow_tts/run-January-13-2023_09+07PM-39a668ff/config.json"

    vocoder_path = "/home/smehta/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan/model_file.pth"
    vocoder_config = "/home/smehta/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan/config.json"

    text = "This is a test sentence."
    model = load_model_components(model_path, config_path)
    vocoder = load_vocoder(vocoder_path, vocoder_config)
    output = synthesise_test(model, vocoder, text)
    wav_form = vocoder.inference(output["model_outputs"].transpose(1, 2))[0].cpu().numpy().T
    vocoder.ap.save_wav(wav_form, "test_glow.wav")
