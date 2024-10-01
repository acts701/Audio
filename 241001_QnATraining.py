import pandas as pd 
from tqdm import tqdm
from sentence_transformers.readers import InputExample
import random
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
import math
import logging
from datetime import datetime
from datasets import load_dataset

dataFile = r'C:/Users/acts7/Documents/GitHub/Audio/data/QnA_Set.csv'
numEpochs = 1
trainBatchSize = 1
# pretrained_model_path = r'C:/Users/acts7/Documents/GitHub/ko-sentence-transformers'
# pretrained_model_path = SentenceTransformer("Huffon/sentence-klue-roberta-base")
pretrained_model_path = "Huffon/sentence-klue-roberta-base"
timestamp = datetime.now()
outModelPath = './model/nli_' + timestamp.strftime("&Y&m&d-%H%M%S")+'/'

def convertNLI(_df):
    return [
        {
            'hypothesis': row['sentence1'],
            'premise': row['sentence2'],
            'label': 0 if row['label'] == 'entailment' 
                     else 1 if row['label'] == 'neutral' 
                     else 2
        }
        for _, row in tqdm(df.iterrows(), total=len(_df))
    ]

def makeNliTripletInputExample(_dict):
    trainData = {}
    def addToSamples(_sentence1, _sentence2, _label):
        if _sentence1 not in trainData:
            trainData[_sentence1] = {'contradiction':set(), 'entailment':set(), 'neutral':set()}
        trainData[_sentence1][_label].add(_sentence2)

    for i, data in enumerate(_dict):
        sentence1 = data['hypothesis'].strip()
        sentence2 = data['premise'].strip()
        if 0 == data['label']:
            label = 'entailment'
        elif 1 == data['label']:
            label = 'neutral'
        else:
            label = 'contradiction'
    
        addToSamples(sentence1, sentence2, label)
        addToSamples(sentence2, sentence1, label)

    inputExamples = []
    for sentence1, others in trainData.items():
        if(len(others['entailment']) > 0 and len(others['contradiction'])) > 0:
            inputExamples.append(InputExample(texts=[sentence1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            inputExamples.append(InputExample(texts=[random.choice(list(others['entailment'])), sentence1, random.choice(list(others['contradiction']))]))
    return inputExamples

if __name__ == '__main__':
    df = pd.read_csv(dataFile, encoding='utf-8')
    dfTrain = df
    dfTrain = df[df['type']=='train']
    dfTrain = convertNLI(dfTrain)
    trainExamples = makeNliTripletInputExample(dfTrain)

    trainDataLoader = NoDuplicatesDataLoader(trainExamples, batch_size=trainBatchSize)
    if not trainDataLoader:
        raise ValueError("Training data is empty.")

    dfValid = df[df['type']=='valid']
    dictValid = convertNLI(dfValid)
    validExamples = []

    scores = {0:0.9, 1:0.5, 2:0.01}
    for row in dictValid:
        score = scores[row['label']]
        validExamples.append(InputExample(texts=[row['hypothesis'], row['premise']], label=score))
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            validExamples, batch_size = trainBatchSize, name = 'sta-dev', show_progress_bar = True,
        )
        
        print('train len = ', len(trainExamples))
        print('valid len = ', len(validExamples))

        embedding_model = models.Transformer(
            model_name_or_path = pretrained_model_path,
            max_seq_length = 256,
            do_lower_case = True
        )

    poolingModel = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,  # 소문자로 변경
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[embedding_model, poolingModel])
    trainLoss = losses.MultipleNegativesRankingLoss(model)
    warmupSteps = math.ceil(len(trainExamples)*numEpochs/trainBatchSize*0.1)
    logging.info('warmup-steps:{}'.format(warmupSteps))
    model.fit(
        train_objectives = [(trainDataLoader, trainLoss)],
        evaluator = evaluator,
        epochs = numEpochs,
        evaluation_steps = 10,
        warmup_steps = warmupSteps,
        save_best_model = True,
        output_path = outModelPath,
        show_progress_bar = True
    )