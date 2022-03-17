#! /bin/bash
set -e

# 1. Download indomain train data
DIR=indomain_train_mrqa
mkdir -p $DIR

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O ./$DIR/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O ./$DIR/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O ./$DIR/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O ./$DIR/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O ./$DIR/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O ./$DIR/NaturalQuestions.jsonl.gz

# 2. Download indomain dev data
DIR=indomain_val_mrqa
mkdir -p $DIR

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O $DIR/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O $DIR/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O $DIR/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O $DIR/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O $DIR/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O $DIR/NaturalQuestions.jsonl.gz

# 3. Download oodomain dev data
DIR=oodomain_val_mrqa
mkdir -p $DIR

wget http://participants-area.bioasq.org/MRQA2019/ -O $DIR/BioASQ.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz -O $DIR/TextbookQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz -O $DIR/RelationExtraction.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz -O $DIR/DROP.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz -O $DIR/DuoRC.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz -O $DIR/RACE.jsonl.gz
