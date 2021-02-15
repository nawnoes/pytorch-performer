from __future__ import absolute_import, division, print_function
# import warnings
# warnings.filterwarnings("ignore")

import logging
import os
import random
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/reformer-language-model')
from io import open
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler,SequentialSampler, TensorDataset)
from tqdm import tqdm, trange,tqdm_notebook

from model.performer import PerformerMLM, PerformerMRCModel
from transformers.optimization import AdamW
from example.schedule import WarmupLinearSchedule
from transformers import BertTokenizer
from example.arg import ModelConfig
from example.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions,evaluate)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
SEED = 42


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model, epoch, eval_examples,eval_features,predict_batch_size):
  predict = dev_file

  all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
  sampler = SequentialSampler(dataset)
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=predict_batch_size)

  logger.info("***** Evaluating *****")
  logger.info("  Num features = %d", len(dataset))
  logger.info("  Batch size = %d", predict_batch_size)

  model.eval()
  model.to(device)

  all_results = []
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)

  logger.info("Start evaluating!")
  for input_ids, input_mask, segment_ids, example_indices in dataloader: #tqdm(dataloader, desc="Evaluating", leave=True, position=1):
    input_ids = input_ids.to(device)

    with torch.no_grad():
      batch_start_logits, batch_end_logits = model(input_ids)

    for i, example_index in enumerate(example_indices):
      start_logits = batch_start_logits[i].detach().cpu().tolist()
      end_logits = batch_end_logits[i].detach().cpu().tolist()
      eval_feature = eval_features[example_index.item()]
      unique_id = int(eval_feature.unique_id)
      all_results.append(RawResult(unique_id=unique_id,
                                   start_logits=start_logits,
                                   end_logits=end_logits))

  output_prediction_file = os.path.join(output_dir, f"{model_name}_predictions_{epoch}.json")
  output_nbest_file = os.path.join(output_dir, f"{model_name}_nbest_predictions_{epoch}.json")
  write_predictions(eval_examples, eval_features, all_results,
                    n_best_size, max_answer_length,
                    False, output_prediction_file, output_nbest_file,
                    None, False, False, 0.0)

  with open(predict) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']

  with open(os.path.join(output_prediction_file)) as prediction_file:
    logger.info(f"Evaluating File: {output_prediction_file}")
    predictions = json.load(prediction_file)
  logger.info(json.dumps(evaluate(dataset, predictions)))


if __name__ == '__main__':

  # google drive path
  # gdrive_path = "/content/drive/My Drive/Colab Notebooks/reformer-language-model"
  gdrive_path = ".."

  # train & eval dataset
  train_file = f"{gdrive_path}/data/korquad/KorQuAD_v1.0_train.json"
  dev_file = f"{gdrive_path}/data/korquad/KorQuAD_v1.0_dev.json"

  # path for train
  output_dir = f'{gdrive_path}/korquad'
  # checkpoint_path = os.path.join(gdrive_path, "checkpoints/reformer-mlm-small.pth") # epoch27-reformer-small.pt
  checkpoint_path = os.path.join(gdrive_path, "checkpoints/performer-mlm-small.pth")
  vocab_path = os.path.join(gdrive_path, "data/wiki-vocab.txt")
  model_name = "performer-mlm-small"

  # Hyperparameter
  doc_stride = 128
  max_query_length = 96
  max_answer_length = 30
  n_best_size = 20

  train_batch_size = 2#64
  learning_rate = 5e-5
  warmup_proportion = 0.1
  num_train_epochs = 10.0

  max_grad_norm = 1.0
  adam_epsilon = 1e-6
  weight_decay = 0.01

  # Device Setting
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  logger.info("device: {} n_gpu: {}".format(device, n_gpu))

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  if n_gpu > 0:
          torch.cuda.manual_seed_all(SEED)

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # Tokenizer
  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

  # 2. Model Hyperparameter
  batch_size = 128
  max_seq_length = 512


  # 3. Prepare model
  model = PerformerMRCModel(
    num_tokens=tokenizer.vocab_size,
    dim=512,
    depth=6,
    heads=8,
    max_seq_len=512,
  )

  # Evaluate data
  eval_examples = read_squad_examples(input_file=dev_file,is_training=False,version_2_with_negative=False)
  eval_features = convert_examples_to_features(examples=eval_examples,tokenizer=tokenizer,max_seq_length=512,doc_stride=doc_stride,max_query_length=max_query_length,is_training=False)

  # 시작 Epoch
  start_epoch = 0

  # MLM 모델
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.performer.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Load Performer Model')

  #######################################################################################################
  # Train
  #######################################################################################################
  # number of model parameter
  num_params = count_parameters(model)
  logger.info("Total Parameter: %d" % num_params)

  model.to(device)

  cached_train_features_file = train_file + '_{0}_{1}_{2}'.format(str(max_seq_length), str(doc_stride),str(max_query_length))
  train_examples = read_squad_examples(input_file=train_file, is_training=True, version_2_with_negative=False)

  try:
    with open(cached_train_features_file, "rb") as reader:
      train_features = pickle.load(reader)
  except:
    train_features = convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=True)
    logger.info("  Saving train features into cached file %s", cached_train_features_file)
    with open(cached_train_features_file, "wb") as writer:
      pickle.dump(train_features, writer)

  num_train_optimization_steps = int(len(train_features) / train_batch_size) * num_train_epochs

  # Prepare optimizer
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = AdamW(optimizer_grouped_parameters,
                    lr=learning_rate,
                    eps=adam_epsilon)
  scheduler = WarmupLinearSchedule(optimizer,
                                   warmup_steps=num_train_optimization_steps * 0.1,
                                   t_total=num_train_optimization_steps)

  logger.info("***** Running training *****")
  logger.info("  Num orig examples = %d", len(train_examples))
  logger.info("  Num split examples = %d", len(train_features))
  logger.info("  Batch size = %d", train_batch_size)
  logger.info("  Num steps = %d", num_train_optimization_steps)
  num_train_step = num_train_optimization_steps

  all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
  all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
  all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_start_positions, all_end_positions)

  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

  model.train()
  global_step = 0
  epoch = start_epoch
  for i in range(start_epoch, int(num_train_epochs)):
    iter_bar = tqdm(train_dataloader, desc=f"Epoch-{i} Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
    tr_step, total_loss, mean_loss = 0, 0., 0.
    for step, batch in enumerate(iter_bar):
      if n_gpu == 1:
        batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

      input_ids, input_mask, segment_ids, start_positions, end_positions = batch

      loss = model(input_ids, start_positions, end_positions)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

      scheduler.step()
      optimizer.step()
      optimizer.zero_grad()
      global_step += 1
      tr_step += 1
      total_loss += loss.item()
      mean_loss = total_loss / tr_step
      iter_bar.set_description(f"Epoch-{i} Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                               (global_step, num_train_step, mean_loss, loss.item()))

    logger.info("** ** * Saving file * ** **")
    model_checkpoint = f"{model_name}_{epoch}.bin"
    logger.info(model_checkpoint)
    output_model_file = os.path.join(output_dir, model_checkpoint)
    # 평가
    evaluate(model, epoch, eval_examples, eval_features, train_batch_size)
    model.train()

    torch.save(model.state_dict(), output_model_file)
    epoch += 1