import util
import uuid
import os
import re
import torch
from tqdm import tqdm
from itertools import starmap

from itertools import islice
from textattack.augmentation import EmbeddingAugmenter, BackTranslationAugmenter
from transformers import MarianTokenizer, MarianMTModel
def custom_back_translate(texts, target_language='de'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print (f"Back Trnaslation Using: {device}")
    target_model_name = 'Helsinki-NLP/opus-mt-en-de'
    target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
    target_model = MarianMTModel.from_pretrained(target_model_name).to(device)
    target_model.eval()

    en_model_name = 'Helsinki-NLP/opus-mt-de-en'
    en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
    en_model = MarianMTModel.from_pretrained(en_model_name).to(device)
    en_model.eval()

    def translate(texts, model, tokenizer, language="fr"):
        with torch.no_grad():
            template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
            src_texts = [template(text) for text in texts]
            encoded = tokenizer.prepare_seq2seq_batch(src_texts,
                                                      truncation=True,
                                                      return_tensors="pt").to(device)
            translated = model.generate(**encoded).to(device)
            translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
            return translated_texts

    def back_translate(texts, source_lang="en", target_lang="fr"):
        # Translate from source to target language
        fr_texts = translate(texts, target_model, target_tokenizer,
                             language=target_lang)
        # Translate from target language back to source language
        back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                          language=source_lang)

        return back_translated_texts

    # batch the input text so that memory doesn't exceed limit
    batch_size  = 16
    result = []
    for start in tqdm(range(0, len(texts), batch_size)):
        result.extend(back_translate(texts[start: start+batch_size], source_lang="en", target_lang=target_language))
    return result

# given some context can be too large. we only back translate the portion around the answer index
def get_context_to_back_translate(context, answer):
    start_index, end_index = len(context), 0
    for i in range(len(answer["text"])):
        start_index = min(start_index, answer["answer_start"][i])
        end_index = max(end_index, answer["answer_start"][i] + len(answer["text"][i]))
    padding = 150 # padd 100 words before and after start and end index
    excerpt_start = max(0, start_index - padding)
    excerpt_end = min(end_index + padding, len(context))
    return context[excerpt_start: excerpt_end]


# given the perturbed data. return the newly encoded data. If the input is not valid (answer text can't be found in context) we will raise an error.
def get_new_data(context, question, answer):
    new_answer = { "text": [], "answer_start": []}

    for answer_text in answer["text"]:
        r = re.search(r'\b%s\b' % answer_text, context)
        if not r:
            print (context)
            raise ValueError("answer text: {} can't be found in context".format(answer_text))
        new_answer["answer_start"].append(r.start())
        new_answer["text"].append(answer_text)

    # generate a random uuid. it's not safe as there might be collision but probably ok.
    return (context, question, new_answer, str(uuid.uuid4()))



def filter_valid_augmentated_data(original_dataset, augmented_context_list, augmented_question_list):
    total_record_sz = len(original_dataset["id"])
    assert (len(augmented_context_list) == len(original_dataset["context"]))
    assert (len(augmented_question_list) == len(original_dataset["question"]))
    for i in range(total_record_sz):
        context, question, answer, old_id = original_dataset['context'][i], original_dataset['question'][i], original_dataset['answer'][i], \
                                            original_dataset['id'][i]
        # first yield the original output:
        yield context, question, answer, old_id

        # Output the augment context
        try:
            yield get_new_data(augmented_context_list[i], question, answer)
        except Exception as e:
            print (e)

        # Output augmented question
        try:
            yield get_new_data(augmented_context_list[i], augmented_question_list[i], answer)
        except Exception as e:
            print (e)

        # change the answer word to a different word
        for transformed_answer in EmbeddingAugmenter(transformations_per_example=1).augment(answer["text"][0]):
            try:
                # we will swap out the transformed words in context.
                transformed_context = re.sub(answer["text"][0], transformed_answer, context)
                yield get_new_data(transformed_context, question, {"text": [transformed_answer]})
            except Exception as e:
                print(e)



def data_set_to_augment(data_set_path):
    data_dict = util.read_squad(data_set_path)
    out_data_path = os.path.join(os.path.dirname(data_set_path), os.path.basename(data_set_path) + "_augmented")

    augmented_data_dict ={'question': [], 'context': [], 'id': [], 'answer': []}
    augmented_context_de = custom_back_translate(list(starmap(get_context_to_back_translate, zip(data_dict["context"], data_dict["answer"]))),
                                                 target_language='de')
    augmented_question_de = custom_back_translate(data_dict["question"], target_language='de')

    for augmented_context, augmented_question, augmented_answer, augmented_id in filter_valid_augmentated_data(data_dict, augmented_context_de, augmented_question_de):
        augmented_data_dict['question'].append(augmented_question)
        augmented_data_dict['context'].append(augmented_context)
        augmented_data_dict['id'].append(augmented_id)
        augmented_data_dict['answer'].append(augmented_answer)

    util.write_squad(out_data_path, augmented_data_dict)

if __name__ == "__main__":
    # text_1 = """
    # n\nNew Orleans, Louisiana, 1927. An enraged posse of men descend on the isolated Seven Doors Hotel deep in the swamps. They grab an artist called Schweik (Antoine Saint John), who is cloistered there. Accusing him of being a warlock, Schweik is dragged down to the cellar where he is savagely beaten with heavy chains, tortured with quicklime acid, and crucified with his wrists nailed to a cellar wall, despite his dire warnings of evil to be unleashed.New Orleans, 1981. Liza Merril (Catriona MacColl) is a young woman who arrives from New York City to claim the hotel as her inheritance. No sooner has architect friend Marin Avery (Michele Mirabella) begins to show her around the property, strange incidents begin to happen. A painter (Anthony Flees) falls off his rig and is horribly injured, coughing up blood and babbling about, \"the eyes, the eyes.\" Dr. John McCabe (David Warbeck) arrives to take the injured man to the hospital, and offers Liza some sympathy. Next, a plumber, named Joe, attempts to repair a major leak in the flooded cellar. However, he is murdered by a presence that emerged from behind a slim-caked wall. The atmosphere at the hotel is further chilled by the creepy-looking servants, Arthur (Giampaolo Saccarola) and Martha (Veronica Lazar), who apparently come with the hotel. Martha discovers Joe's dead body in the cellar, and another much older cadaver lying in a pool of dirty water nearby. It is apparently that of Schweik, the artist.Driving down the 14-mile causeway to New Orleans, Liza encounters a strange blind woman, standing in the middle of the desolate highway. The blind woman introduces herself as Emily (Sarah Keller), and tells Liza that she has been waiting for her, although her eyes are occluded with cataracts. Liza drives Emily over to her opulently furnished house in New Orleans. Liza is warned by Emily to leave the hotel while she still can. Meanwhile at the hospital morgue, Dr. John McCabe is performing the autopsy on Joe the plumber while his assistant Harris (Al Cliver) wants to install an EMG machine to the corpse of Schweik. John laughs it off and leaves for lunch, while Harris remains behind to install the EMG machine. After Harris leaves for a call, the EMG machine begins pulsing with activity. A little later, Joe's wife Mary-Anne (Laura De Marchi) arrives with her daughter Jill (Maria Pia Marsale) to dress up her husband's corpse for the funeral, when she is killed in a horrific way by scalded with acid. Jill is then menaced by the re-animated cadaver of Schweik.Liza meets with John McCabe in a downtown bar to discuss her misgivings and anxieties.
    # """
    #
    # text_2 = """
    # i hate you so much stupid
    # """
    #
    # print(custom_back_translate([text_1, text_2]))
    data_set_to_augment("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc")
    # data_set_to_augment("/home/kaiyuewang/QAGAN/datasets/oodomain_train/race")
    # data_set_to_augment("/home/kaiyuewang/QAGAN/datasets/oodomain_train/relation_extraction")

    # original_dict = util.read_squad("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc")
    # new_dict = util.read_squad("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc_augmented")
    # import json
    # print (json.dumps(original_dict, sort_keys=True) == json.dumps(new_dict, sort_keys=True))

