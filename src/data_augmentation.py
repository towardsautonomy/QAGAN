import util
import uuid
import os
import re
from textattack.augmentation import EmbeddingAugmenter, BackTranslationAugmenter

# given the perturbed data. return the newly encoded data. The input should be garenteed to be valid.
def get_new_data(context, question, answer):
    new_answer = { "text": [], "answer_start": []}

    for answer_text in answer["text"]:
        r = re.search(r'\b%s\b' % answer_text, context)
        if not r:
            raise ValueError("answer text: {} can't be found in context".format(answer_text))
        new_answer["answer_start"].append(r.start())
        new_answer["text"].append(answer_text)

    # generate a random uuid. it's not safe as there might be collision but probably ok.
    return (context, question, answer, uuid.uuid4())


def augment_data(data_dict, i):
    context, question, answer, old_id = data_dict['context'][i], data_dict['question'][i], data_dict['answer'][i], data_dict['id'][i]
    back_translate_augmenter = BackTranslationAugmenter(transformations_per_example=1)
    embedding_augmenter = EmbeddingAugmenter(transformations_per_example=5)
    print ("Augmenting: {}".format(context))
    # try paraphrasing the context first. we would only output the data if after paraprasing the answer text still exists in the paraprased context

    # try paraphrasing the question. We don't do extra check here and assume that question make sense.
    for transformed_question in BackTranslationAugmenter(transformations_per_example=3).augment(question):
        try:
            yield get_new_data(context, transformed_question, answer)
        except ValueError as e:
            print (e)

    # transform the context by swaping out words with similar word close in embedding
    for transformed_context in EmbeddingAugmenter(transformations_per_example=5).augment(context):
        try:
            yield get_new_data(transformed_context, question, answer)
        except ValueError as e:
            print (e)

    # change the answer word to a different word
    # for transformed_answer in EmbeddingAugmenter(transformations_per_example=5).augment(answer["text"][0]):
        # transformed_context = re.sub(answer["text"][0], transformed_answer, context)
        yield get_new_data(transformed_context, question, answer)

    for transformed_context in BackTranslationAugmenter(transformations_per_example=1).augment(context):
        try:
            yield get_new_data(transformed_context, question, answer)
        except ValueError as e:
            print (e)

def data_set_to_augment(data_set_path):
    data_dict = util.read_squad(data_set_path)
    out_data_path = os.path.join(os.path.dirname(data_set_path), os.path.basename(data_set_path) + "_augmented")

    augmented_data_dict ={'question': [], 'context': [], 'id': [], 'answer': []}

    total_record_sz = len(data_dict["id"])
    for i in range(total_record_sz):
        for augmented_context, augmented_question, augmented_answer, augmented_id in augment_data(data_dict, i):
            augmented_data_dict['question'].append(augmented_question)
            augmented_data_dict['context'].append(augmented_context)
            augmented_data_dict['id'].append(augmented_id)
            augmented_data_dict['answer'].append(augmented_answer)

    util.write_squad(out_data_path, augmented_data_dict)

if __name__ == "__main__":
    data_set_to_augment("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc")
    original_dict = util.read_squad("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc")
    new_dict = util.read_squad("/home/kaiyuewang/QAGAN/datasets/oodomain_train/duorc_augmented")
    import json
    print (json.dumps(original_dict, sort_keys=True) == json.dumps(new_dict, sort_keys=True))

