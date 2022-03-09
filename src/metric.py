from typing import List
from textattack.metrics import Perplexity
from textattack.augmentation.augmenter import AugmentationResult
import torch
from torch.nn.functional import cross_entropy
class data:
    def __init__(self, text):
        self.text = text

def get_perplexity(texts: List[str], ppl_model, ppl_tokenizer):
    max_length = ppl_model.config.n_positions
    stride = 512
    # cut text into segment with stride
    # def text_to_smaller_text(text):
    #     return [text[i: min(i + stride, len(text))] for i in range(0, len(text), stride)]
    # texts = map(text_to_smaller_text, texts)
    #
    tokens = [ppl_tokenizer.convert_tokens_to_ids(
        ppl_tokenizer.tokenize(x, add_prefix_space=True))
        for x in texts]
    tokens = [[ppl_tokenizer.bos_token_id] + x + [ppl_tokenizer.eos_token_id]
              for x in tokens]
    inputs = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x) for x in tokens], batch_first=True, padding_value=0).to('cuda')
    mask = (inputs != 0).float()
    # replace the ids of the padded tokens (where token_id==padded_id) with `-1`
    labels = inputs.masked_fill(inputs == 0, 0)
    with torch.no_grad():
        outputs = ppl_model(inputs, attention_mask=mask, labels=labels)
    loss, logits = outputs[:2]
    logits_= torch.transpose(logits, 1, 2) # (Num sample, Dimension, Num words)
    _labels = torch.cat([inputs[:, 1:], inputs[:, :1] * 0], dim=1)
    loss_real = cross_entropy(logits_, _labels, ignore_index=0,
                                    reduction='none') * mask # ignore padded timesteps

    return torch.sum(loss_real, dim=1) / mask.sum(dim=1)  # normalize by sentence length

def calculate_perplexity(original_text: List[str],  translated_text: List[str]):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    try:
        ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
        ppl_model.to("cuda")
        ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        ppl_model.eval()
        original_ppl = get_perplexity(original_text, ppl_model, ppl_tokenizer)
        # print("Original: ", torch.exp(original_ppl).tolist())
        translated_ppl = get_perplexity(translated_text, ppl_model, ppl_tokenizer)
        # print ("Translated: ", torch.exp(translated_ppl).tolist())
        return torch.exp(original_ppl).tolist(), torch.exp(translated_ppl).tolist()
    except Exception as e:
        return [None] * len(original_text), [None] * len(translated_text)

def get_perplexity_data(original_text, tranlated_text):
#     return calculate_perplexity(original_text, tranlated_text)
    return Perplexity().calculate([AugmentationResult(data(original_text), data(tranlated_text))])

if __name__ == "__main__":
    calculate_perplexity(["where there is a will there is a way", "Towns along the Texas shoreline were inundated, and Galveston Island was devastated, with nearly every building washed away"], ["you is so stupid man", "la is duo kompu kdf lkd ieu calud kds"])