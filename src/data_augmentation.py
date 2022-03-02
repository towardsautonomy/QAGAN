import util
import uuid
from textattack.augmentation import EmbeddingAugmenter, BackTranslationAugmenter

def augment_data(data_dict, i):
    context, question, answer = data_dict['context'][i], data_dict['question'][i], data_dict['answer'][i]
    id = uuid.uuid4() # generate a random uuid. it's not safe as there might be collision but probably ok.

def data_set_to_augment(data_set_path):
    data_dict = util.read_squad(data_set_path)
    augmented_data_dict ={'question': [], 'context': [], 'id': [], 'answer': []}

    total_record_sz = len(data_dict["id"])
    for i in range(total_record_sz):
        for augmented_context, augmented_question, augmented_answer, augmented_id in augment_data(data_dict, i):
            augmented_data_dict['question'].append(augmented_question)
            augmented_data_dict['context'].append(augmented_context)
            augmented_data_dict['id'].append(augmented_id)
            augmented_data_dict['answer'].append(augmented_answer)
    return augmented_data_dict


