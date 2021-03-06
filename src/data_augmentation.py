import util
import uuid
import os
import re
import torch
from tqdm import tqdm
from itertools import starmap
import argparse
from itertools import islice
from textattack.augmentation import EmbeddingAugmenter, BackTranslationAugmenter
from transformers import MarianTokenizer, MarianMTModel
from metric import calculate_perplexity, get_perplexity_data

def custom_back_translate(texts, target_language='de', use_fast_metric=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print (f"Back Trnaslation Using: {device}")
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
    perplexity = []
    perpleixty_true_text = []
    perpleixty_translated_text = []
    # print ("length of current batch: ", map(len, texts))
    for start in tqdm(range(0, len(texts), batch_size)):
        translated = back_translate(texts[start: start+batch_size], source_lang="en", target_lang=target_language)
        result.extend(translated)
        if not use_fast_metric:
            perplexity.extend(list(starmap(get_perplexity_data, zip(texts[start: start+batch_size], translated))))
        else:
            perplexity_true, perplexity_translated = calculate_perplexity(texts[start: start + batch_size], translated)
            perplexity.extend([{"original_text": orig, "translated_text": translated} for orig, translated in zip(perplexity_true, perplexity_translated)])
    return result, perplexity

# given some context can be too large. we only back translate the portion around the answer index
def get_context_to_back_translate(context, answer):
    start_index, end_index = len(context), 0
    for i in range(len(answer["text"])):
        start_index = min(start_index, answer["answer_start"][i])
        end_index = max(end_index, answer["answer_start"][i] + len(answer["text"][i]))
    padding = 150 # padd 150 words before and after start and end index
    extra_start_padding, extra_end_padding = 0, 0
    maximum_extra = 100
    sentence_end_char = ['.', '?', '!']
    excerpt_start = max(0, start_index - padding)
    excerpt_end = min(end_index + padding, len(context))

    while excerpt_start > 0 and context[excerpt_start] not in sentence_end_char and extra_start_padding < maximum_extra:
        excerpt_start -= 1
        extra_start_padding += 1

    while excerpt_end < len(context) and context[excerpt_end] not in sentence_end_char and extra_end_padding < maximum_extra:
        excerpt_end += 1
        extra_end_padding += 1

    return context[excerpt_start: excerpt_end]


# given the perturbed data. return the newly encoded data. If the input is not valid (answer text can't be found in context) we will raise an error.
def get_new_data(context, question, answer):
    new_answer = { "text": [], "answer_start": []}

    for answer_text in answer["text"]:
        r = re.search(r'\b%s\b' % answer_text.lower(), context.lower())
        if not r:
            raise ValueError("answer text: {} can't be found in context".format(answer_text))
        new_answer["answer_start"].append(r.start())
        new_answer["text"].append(answer_text.lower())

    # generate a random uuid. it's not safe as there might be collision but probably ok.
    return (context, question, new_answer, str(uuid.uuid4()))

def filter_valid_augmentated_data(original_dataset, augmented_context_list, augmented_question_list, context_perplexity, question_perplexity):
    total_record_sz = len(original_dataset["id"])
    assert (len(augmented_context_list) == len(original_dataset["context"]))
    assert (len(augmented_question_list) == len(original_dataset["question"]))
    for i in range(total_record_sz):
        context, question, answer, old_id = original_dataset['context'][i], original_dataset['question'][i], original_dataset['answer'][i], \
                                            original_dataset['id'][i]
        # first yield the original output:
        yield context, question, answer, old_id, None, None

        # Output the augment context
        try:
            new_context, new_question, new_answer, new_id = get_new_data(augmented_context_list[i], question, answer)

            yield new_context, new_question, new_answer, new_id, context_perplexity[i], None
        except Exception as e:
            print (e)

        # Output augmented question
        try:
            new_context, new_question, new_answer, new_id = get_new_data(context, augmented_question_list[i], answer)
            yield new_context, new_question, new_answer, new_id, None, question_perplexity[i]

        except Exception as e:
            print (e)

        # change the answer word to a different word
        # for transformed_answer in EmbeddingAugmenter(transformations_per_example=1).augment(answer["text"][0]):
        #     try:
        #         # we will swap out the transformed words in context.
        #         transformed_context = re.sub(answer["text"][0], transformed_answer, context)
        #         yield get_new_data(transformed_context, question, {"text": [transformed_answer]})
        #     except Exception as e:
        #         print(e)



def data_set_to_augment(data_set_path, use_fast_metric=False):
    data_dict = util.read_squad(data_set_path)
    out_data_path = os.path.join(os.path.dirname(data_set_path), os.path.basename(data_set_path) + "_augmented")

    augmented_data_dict ={'question': [], 'context': [], 'id': [], 'answer': [], 'context_perplexity': [], 'question_perplexity': []}
    augmented_context_de, context_perplexity = custom_back_translate(list(starmap(get_context_to_back_translate, zip(data_dict["context"], data_dict["answer"]))),
                                                 target_language='de', use_fast_metric=use_fast_metric)
    augmented_question_de, question_perplexity = custom_back_translate(data_dict["question"], target_language='de', use_fast_metric=use_fast_metric)

    for augmented_context, augmented_question, augmented_answer, augmented_id, context_perplexity, question_perplexity in filter_valid_augmentated_data(data_dict, augmented_context_de, augmented_question_de, context_perplexity, question_perplexity):
        augmented_data_dict['question'].append(augmented_question)
        augmented_data_dict['context'].append(augmented_context)
        augmented_data_dict['id'].append(augmented_id)
        augmented_data_dict['answer'].append(augmented_answer)
        augmented_data_dict['context_perplexity'].append(context_perplexity)
        augmented_data_dict['question_perplexity'].append(question_perplexity)

    util.write_squad(out_data_path, augmented_data_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--domain', type=str, default="indomain_train",
                        help='string name for domain directory (indomain_train|indomain_val|oodomain_train|etc')
    parser.add_argument('--datasets', type=str, default="duorc",
                        help='list of questions nat_questions_augmented newsqa_augmented connected by comma')
    parser.add_argument('--fast_metric', type=bool, default="false",
                        help='whether to use fast perplexity computation')

    args = parser.parse_args()
    for dataset in args.datasets.split(','):
        data_set_to_augment(os.path.join(os.path.dirname(__file__), "..", "datasets", args.domain, dataset), args.fast_metric)

