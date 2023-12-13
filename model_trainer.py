from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import requests
import csv


def getGoogleSeet(spreadsheet_id, outDir, outFile):
    url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv'
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(outDir, outFile)
        with open(filepath, 'wb') as f:
            f.write(response.content)
            print('CSV file saved to: {}'.format(filepath))
            return filepath
    else:
        print(f'Error downloading Google Sheet: {response.status_code}')


outDir = 'data/'
os.makedirs(outDir, exist_ok=True)
sts_dataset_path = getGoogleSeet('12Y-3dwI43jTe-teXGF3cfKxz1QNefXHQj3v3NJhJBho', outDir, "sts-train.csv")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

train_batch_size = 16
num_epochs = 8
model_save_path = 'model/trained-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model = CrossEncoder('sentence-transformers/sentence-t5-base', num_labels=1)

logger.info("Read STS-train dataset")

train_samples = []
dev_samples = []
test_samples = []
with open(sts_dataset_path, 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        else:
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            train_samples.append(InputExample(texts=[row['sentence2'], row['sentence1']], label=score))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logger.info("Warmup-steps: {}".format(warmup_steps))

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

model = CrossEncoder(model_save_path)

evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
evaluator(model)
