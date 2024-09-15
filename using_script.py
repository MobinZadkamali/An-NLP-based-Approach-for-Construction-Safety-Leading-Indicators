from models.transformer.seqTagger import transformertagger
from models.transformer.BERT_sentence_classification import Bert
import torch

if __name__ == '__main__':

    # now we use it
    save_path = 'test_bert'

    tagger_obj = transformertagger(save_path, Bert, {1: 0.0, 2: 1.0}, device=torch.device("cpu"))

    toks, intents = tagger_obj.get_label(["تهیه، ساخت و نصب قاب‌ها (تا دهانه 30 متر)، که جان و بال آنها از ورق بریده و ساخته شده‌اند (با ارتفاع جان متغیر)، با کف ستون‌ها، انواع ورقه‌های اتصالی، تقویتی و اتصاال‌های واسطه با پیچ و مهره، همراه با برشکاری، سوراخکاری، جوشکاری و سابیدن."], need_tokenization=True)

    for i in range(len(toks)):
      print("\n## Intent:", intents[i])

