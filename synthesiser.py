import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from hifigan.env import AttrDict
from hifigan.models import Generator
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def load_hifigan(device):
    hifigan_loc = Path('hifigan')
    config_file = hifigan_loc / 'config_v1.json'
    hifi_checkpoint_file = hifigan_loc / 'g_02500000'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(hifi_checkpoint_file, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


#


# 83059180
test_sentences = {
    1: "The position of this palmprint on the carton was parallel with the long axis of the box and at right angles with the short axis.",
    2: "We will not allow ourselves to run around in new circles of futile discussion and debate, always postponing the day of decision.",
    3: "It is now possible for Presidents to travel the length and breadth of a land far larger than the United States.",
    4: "This is proved by contemporary accounts, especially one graphic and realistic article which appeared in the 'Times'.",
    5: "A formal and thorough description of the responsibilities of the advance agent is now in preparation by the Service.",
    6: "The rentals of rooms and fees went to the warden, whose income was two thousand three hundred seventy two pounds.",
    7: "With respect to industry and business, but nearly all are agreed that private enterprise in times such as these.",
    8: "The only exit from the office in the direction Oswald was moving was through the door to the front stairway.",
    9: "it was the beginning of the great collections of Babylonian antiquities in the museums of the Western world.",
    10: "The bank enjoyed an excellent reputation, it had a good connection, and was supposed to be perfectly sound.",
    11: "But he did not like the risk of entering a room by the fireplace, and the chances of detection it offered.",
    12: "Later on, he had devoted himself to the personal investigation of the prisons of the United States.",
    13: "And the windows of the opposite houses, which commanded a good view, as usual fetched high prices.",
    14: "The boy declared he saw no one, and accordingly passed through without paying the toll of a penny.",
    15: "There should be at least a full and fair trial given to these means of ending industrial warfare.",
    16: "Many years were to elapse before these objections should be fairly met and universally overcome.",
    17: "Under the new system the whole of the arrangements from first to last fell upon the officers.",
    18: "And in many directions, the intervention of that organized control which we call government",
    19: "and in such an effort we should be able to secure for employers and employees and consumers.",
    20: "Nor did the methods by which they were perpetrated greatly vary from those in times past.",
    21: "The narrative of these events is based largely on the recollections of the participants.",
    22: "It had established periodic regular review of the status of four hundred individuals.",
    23: "He had repeated this wish only a few days before, during his visit to Tampa, Florida.",
    24: "His disclaimer, distinct and detailed on every point, was intended simply for effect.",
    25: "I then stepped off of it and the officer picked it up in the middle and it bent so.",
    26: "That is reflected in definite and comprehensive operating procedures.",
    27: "The second step we have taken in the restoration of normal business enterprise.",
    28: "In all these lines the facts are drawn together by a strong thread of unity.",
    29: "Yet the public opinion of the whole body seems to have checked dissipation.",
    30: "The jury did not believe him, and the verdict was for the defendants.",
    31: "One by one the huge uprights of black timber were fitted together.",
    32: "He was a tall, slender man, with a long face and iron gray hair.",
    33: "He had been committed for an offense for which he was acquitted.",
    34: "Three others subsequently identified Oswald from a photograph.",
    35: "The eyewitnesses vary in their identification of the jacket.",
    36: "High authorities were in favor of continuous separation.",
    37: "Oswald rose from his seat, bringing up both hands.",
    38: "But there were other notorious cases of forgery.",
    39: "Yet the law was seldom if ever enforced.",
    40: "This was all the police wanted to know.",
}



lj_valid = [
    "The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read.",
    "If somebody did that to me, a lousy trick like that, to take my wife away, and all the furniture, I would be mad as hell, too.",
    "as is shown by the report of the Commissioners to inquire into the state of the municipal corporations in 1835.",
    "Even the Caslon type when enlarged shows great shortcomings in this respect:",
    "All the committee could do in this respect was to throw the responsibility on others.",
    "These pungent and well grounded strictures applied with still greater force to the unconvicted prisoner, the man who came to the prison innocent, and still uncontaminated.",
    "and recognized as one of the frequenters of the bogus law stationers. His arrest led to that of others.",
    "Oswald was, however, willing to discuss his contacts with Soviet authorities. He denied having any involvement with Soviet intelligence agencies",
    "The first physician to see the President at Parkland Hospital was Dr. Charles J. Carrico, a resident in general surgery.",
    "during the morning of November 22 prior to the motorcade.",
    "On occasion the Secret Service has been permitted to have an agent riding in the passenger compartment with the President.",
    "although at Mr. Buxton's visit a new jail was in process of erection, the first step towards reform since Howard's visitation in 1774.",
    "or theirs might be one of many, and it might be considered necessary to make an example.",
    "The Warren Commission Report. By The President's Commission on the Assassination of President Kennedy. Chapter 7. Lee Harvey Oswald.",
    "Mr. Wakefield winds up his graphic but somewhat sensational account by describing another religious service, which may appropriately be inserted here.",
    "A modern artist would have difficulty in doing such accurate work.",
    "with the particular purposes of the agency involved. The Commission recognizes that this is a controversial area",
    "Oswald's Marine training in marksmanship, his other rifle experience and his established familiarity with this particular weapon",
    "According to O'Donnell, quote, we had a motorcade wherever we went, end quote.",
    "Dr. Clark, who most closely observed the head wound.",
    "Euins, who was on the southwest corner of Elm and Houston Streets testified that he could not describe the man he saw in the window.",
    "Energy enters the plant, to a small extent.",
    "once you know that you must put the cross hairs on the target and that is all that is necessary.",
    "the fatal consequences whereof might be prevented if the justices of the peace were duly authorized",
    "Speaking on a debate on prison matters, he declared that",
    "he was reported to have fallen away to a shadow.",
    "His disappearance gave color and substance to evil reports already in circulation that the will and conveyance above referred to",
    "Here the tread wheel was in use, there cellular cranks, or hard labor machines.",
    "you tap gently with your heel upon the shoulder of the dromedary to urge her on.",
    "This plan of mine is no attack on the Court;",
    "No nightclubs or bowling alleys, no places of recreation except the trade union dances. I have had enough.",
    "The police asked him whether he could pick out his passenger from the lineup.",
    "During his Presidency, Franklin D. Roosevelt made almost four hundred journeys and traveled more than three hundred fifty thousand miles.",
    "He was seen afterwards smoking and talking with his hosts in their back parlor, and never seen again alive.",
    "long narrow rooms, one 36 feet, six 23 feet, and the eighth 18.",
    "We come to the sermon.",
    "even when the high sheriff had told him there was no possibility of a reprieve, and within a few hours of execution.",
    "but there is a system for the immediate notification of the Secret Service by the confining institution when a subject is released or escapes.",
    "When other pleasures palled he took a theater, and posed as a munificent patron of the dramatic art.",
    "Old exchange rate in addition to his factory salary of approximately equal amount",
    "Hill had both feet on the car and was climbing aboard to assist President and Mrs. Kennedy.",
    "seeing that since the establishment of the Central Criminal Court, Newgate received prisoners for trial from several counties.",
    "then let twenty days pass, and at the end of that time station near the Chaldasan gates a body of four thousand.",
    "While they were in a state of insensibility the murder was committed.",
    "reached the same conclusion as Latona that the prints found on the cartons were those of Lee Harvey Oswald.",
    "These were damnatory facts which well supported the prosecution.",
    "but were the precautions too minute, the vigilance too close to be eluded or overcome?",
    "but his scribe wrote it in the manner customary for the scribes of those days to write of their royal masters.",
    "The inadequacy of the jail was noticed and reported upon again and again by the grand juries of the city of London.",
    "At last, in the twentieth month.",
    "which he kept concealed in a hiding place with a trap door just under his bed.",
    "He married a lady also belonging to the Society of Friends, who brought him a large fortune, which, and his own money, he put into a city firm.",
    "Roger D. Craig, a deputy sheriff of Dallas County.",
    "Other officials, great lawyers, governors of prisons, and chaplains supported this view.",
    "who came from his room ready dressed, a suspicious circumstance, as he was always late in the morning.",
    "is closely reproduced in the life history of existing deer. Or, in other words.",
    "accordingly they committed to him the command of their whole army, and put the keys of their city into his hands.",
    "Mrs. Kennedy chose the hospital in Bethesda for the autopsy because the President had served in the Navy.",
    "From those willing to join in establishing this hoped for period of peace.",
    "Muller, Muller, He's the man, till a diversion was created by the appearance of the gallows, which was received with continuous yells.",
    "Years later, when the archaeologists could readily distinguish the false from the true.",
    "his defense being that he had intended to commit suicide, but that, on the appearance of this officer who had wronged him.",
    "together with a great increase in the payrolls, there has come a substantial rise in the total of industrial profits",
    "After this the sheriffs sent for another rope, but the spectators interfered, and the man was carried back to jail.",
    "and improve the morals of the prisoners, and shall insure the proper measure of punishment to convicted offenders.",
    "drove to the northwest corner of Elm and Houston, and parked approximately 10 feet from the traffic signal.",
    "This is the approximate time he entered the rooming house, according to Earlene Roberts, the housekeeper there.",
    "The criteria in effect prior to November 22, 1963, for determining whether to accept material for the PRS general files",
    "and the deepest anxiety was felt that the crime, if crime there had been, should be brought home to its perpetrator.",
    "but his sporting operations did not prosper, and he became a needy man, always driven to desperate straits for cash.",
    "He was soon afterwards arrested on suspicion, and a search of his lodgings brought to light several garments saturated with blood;",
    "He never reached the cistern, but fell back into the yard, injuring his legs severely.",
    "when he was finally apprehended in the Texas Theatre. Although it is not fully corroborated by others who were present.",
    "and she must have run down the stairs ahead of Oswald and would probably have seen or heard him.",
    "afterwards express a wish to murder the Recorder for having kept them so long in suspense.",
    "nearly indefinitely deferred.",
    "On October 25.",
    "They entered a stone cold room, and were presently joined by the prisoner.",
    "that he could only testify with certainty that the print was less than three days old.",
    "Mrs. Mary Brock, the wife of a mechanic who worked at the station, was there at the time and she saw a white male.",
    "Chapter 7. Lee Harvey Oswald Background and Possible Motives, Part 1.",
    "The arguments he used to justify his use of the alias suggest that Oswald may have come to think that the whole world was becoming involved",
    "the number and names on watches, were carefully removed or obliterated after the goods passed out of his hands.",
    "On the 7th July, 1837.",
    "contracted with sheriffs and conveners to work by the job.",
    "at a distance from the prison.",
    "These principles of homology are essential to a correct interpretation of the facts of morphology.",
    "On one occasion Mrs. Johnson, accompanied by two Secret Service agents, left the room to see Mrs. Kennedy and Mrs. Connally.",
    "which Sir Joshua Jebb told the committee he considered the proper elements of penal discipline.",
    "At the first the boxes were impounded, opened, and found to contain many of O'Connor's effects.",
    "on Brennan's subsequent certain identification of Lee Harvey Oswald as the man he saw fire the rifle.",
    "11. If I am alive and taken prisoner.",
    "yet he could not overcome the strange fascination it had for him, and remained by the side of the corpse till the stretcher came.",
    'I noticed when I went out that the light was on".',
    "He was never satisfied with anything.",
    "and others who were present say that no agent was inebriated or acted improperly.",
    'He was in consequence put out of the protection of their internal law ". Their code was a subject of some curiosity.',
    "Let me retrace my steps, and speak more in detail of the treatment of the condemned in those bloodthirsty and brutally indifferent days.",
    "The original plan called for the President to spend only one day in the State, making whirlwind visits to Dallas, Fort Worth, San Antonio, and Houston.",
    "Mr. Sturges Bourne, Sir James Mackintosh, Sir James Scarlett, and William Wilberforce.",
]


lj_valid = list(test_sentences.values())

# test_sentences ={ 1 : "THE DIFFERENCE IN THE RAINBOW DEPENDS CONSIDERABLY UPON THE SIZE OF THE DROPS." }




MODELS = {
    # "glow": "tts_models/en/ljspeech/glow-tts",
    # "tacotron2": "tts_models/en/ljspeech/tacotron2-DCA",
    "overflow": "tts_models/en/ljspeech/overflow",
    # "FastPitch": "tts_models/en/ljspeech/fast_pitch",
    # "vits": "tts_models/en/ljspeech/vits",

}

MODELS_PATH = {
        "overflow": {
            "model_path": "recipes/ljspeech/overflow/berzlius/checkpoint_{}.pth",
            "config_path": "recipes/ljspeech/overflow/berzlius/config.json",
            "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2"
        },
    # "glow": {
        # "model_path": "recipes/ljspeech/glow_tts/run-January-13-2023_09+07PM-39a668ff/checkpoint_100000.pth",
        # "config_path": "recipes/ljspeech/glow_tts/run-January-13-2023_09+07PM-39a668ff/config.json",
        # "vocoder_name": "vocoder_models/en/ljspeech/multiband-melgan"
    #     "model_path": "~/.local/share/tts/tts_models--en--ljspeech--glow-tts/model_file.pth",
    #     "config_path": "~/.local/share/tts/tts_models--en--ljspeech--glow-tts/config.json",
    #     "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2"
    # },
    # "tacotron2": {
    #     "model_path": "~/.local/share/tts/tts_models--en--ljspeech--tacotron2-DCA/model_file.pth",
    #     "config_path": "~/.local/share/tts/tts_models--en--ljspeech--tacotron2-DCA/config.json",
    #     "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2"
    # }
    # "FastPitch": {
    #     "model_path": "recipes/ljspeech/fast_pitch/fast_pitch_ljspeech-January-15-2023_09+35AM-bd402331/checkpoint_{}.pth",
    #     "config_path": "recipes/ljspeech/fast_pitch/fast_pitch_ljspeech-January-15-2023_09+35AM-bd402331/config.json",
    #     "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2"
    # },
    # "vits": {
    #     "model_path": "recipes/ljspeech/vits_tts/vits_ljspeech-January-17-2023_02+23PM-2a958722/checkpoint_{}.pth",
    #     "config_path": "recipes/ljspeech/vits_tts/vits_ljspeech-January-17-2023_02+23PM-2a958722/config.json",
    #     "vocoder_name": None
    # }
}

# iterations = [500, 1000, 1500, 2000, 2500, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
# iterations = [120000, 140000, 160000, 180000, 190000]
iterations = [40000]

manager = ModelManager()

FOLDER = Path("synth_output_lj_valid") / 'FastPitch_extra'
FOLDER.mkdir(exist_ok=True, parents=True)
generator = load_hifigan("cuda")        

for model_name, model in MODELS.items():
    for iteration in iterations:
        if model_name in MODELS_PATH:
            model_path = MODELS_PATH[model_name]["model_path"].format(iteration)
            config_path = MODELS_PATH[model_name]["config_path"]
            model_item = {}
            model_item["default_vocoder"] = MODELS_PATH[model_name]["vocoder_name"]
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
            my_vocoder=generator if vocoder_name is not None else None,
            use_cuda=True,
        )
        
        # pbar = tqdm(test_sentences.items())
        pbar = tqdm(enumerate(lj_valid))
        for idx, text in pbar:
            idx += 1
            pbar.set_description(f"{model_name}-{idx}-{iteration}")
            wav = synthesiser.tts(text, split=False)
            save_folder = FOLDER / model_name / str(iteration)
            save_folder.mkdir(exist_ok=True, parents=True)
            synthesiser.save_wav(wav, save_folder / f"{model_name}_{idx}.wav")
