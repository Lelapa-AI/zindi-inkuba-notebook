import csv
from sklearn.metrics import f1_score, accuracy_score
from typing import List
import numpy as np
import pandas as pd
from statistics import mean
import re
import random
import evaluate
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List

def zindi_score(submission_file):
  df = pd.read_csv(submission_file)
  avg_score = df["Log-Likelihood"].mean()
  size = df[df['Instruction']=='Model size']['Input Text']
  score = (avg_score + (1-(size/232025))*avg_score)/2 #wat 
  return score

def calculate_chrf(df):
  chrf_metric = evaluate.load("chrf")
  references=format_references(df['Targets'].to_list())
  predictions=df['Response'].to_list()
  score = chrf_metric.compute(predictions=predictions, references=references)
  return score

def format_references(list_of_references: List) -> List:
    """
    This function formats the references in a List of List, as expected by the CHRF metric (and BLEU as well)
    """
    return [[reference] for reference in list_of_references]

def filter_predictions(prediction_df, task):
  # check if instruction is in prediction
  prediction_df['Response'] = prediction_df['Response'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()
  prediction_df['Response'] = prediction_df.apply(remove_instruction_from_pred, axis=1)
   # normalise text
  prediction_df = prediction_df.fillna("unknown")
  prediction_df[['Targets', 'Response']] = prediction_df[['Targets', 'Response']].applymap(normalize_text)
  if task == 'mt':
    return prediction_df

  # extract values based on task keyword (this needs to be updated)
  prediction_df['Response'] = prediction_df['Response'].apply(verbalizer)
  if task == "xnli":
    replacements = {"positive": "entailment", "negative": "contradiction"}
    prediction_df['Response'] = prediction_df['Response'].replace(replacements)
  prediction_df['Response'] = prediction_df.apply(assign_label, axis=1, task=task)

  return prediction_df

def loglikelihoods_to_predictions(log_likelihoods: List[float], labels: List[str]) -> List[str]:
    predictions = []

    for likelihoods in log_likelihoods:
        predicted_label_idx = np.argmax(likelihoods)
        predictions.append(labels[predicted_label_idx])

    return predictions

def compute_f1_and_accuracy(predictions: List[str], ground_truths: List[str], labels: List[str]) -> (float, float):
    label_to_id = {label: i for i, label in enumerate(labels)}

    y_true = [label_to_id[label] for label in ground_truths]
    y_pred = [label_to_id[label] for label in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, f1

def main(csv_file_path):
    log_likelihoods = []
    ground_truths = []

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
          try:
            task = row['Task']
          except:
            task = "xnli"
          lang = row['Langs']
          if task == "sentiment" and lang == 'swahili':
            labels=["Chanya","Wastani","Hasi"]
          elif task == "sentiment" and lang == 'hausa':
            labels=["Kyakkyawa","Tsaka-tsaki","Korau"]
          else:
            labels=["0","1","2"]
          likelihood = [float(x) for x in row['Log-Likelihood'].strip('[]').split(',')]
          log_likelihoods.append(likelihood)
          ground_truths.append(row['Targets'])

    predictions = loglikelihoods_to_predictions(log_likelihoods, labels)
    accuracy, f1 = compute_f1_and_accuracy(predictions, ground_truths, labels)
    print(f"Accuracy: {accuracy}, F1 Score: {f1}")

def do_eval_compute(df, labels):
  """
  This function gets the accuracy and F1 score for a task
  given the labels for the task, the logliklihoods and
  targets
  """
  log_likelihoods = df['Log-Likelihood'].apply(lambda x: float(x).strip('[]').split(','))
  ground_truths = df['Targets']
  predictions = loglikelihoods_to_predictions(log_likelihoods, labels)
  accuracy, f1 = compute_f1_and_accuracy(predictions, ground_truths, labels)
  return f1

def evaluate_zindi(df):
  """
  This function is the same as function above, it just does
  things in memory so the function can be used for the final eval
  as well.

  First step is to separate df such that each task has its own accuracy calc
  """
  scores = []

  df_temp = df[df['Task'] == 'sent'& df['Langs']=='hausa']
  labels = ["Kyakkyawa","Tsaka-tsaki","Korau"]
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for Sentiment Hausa:", res)

  df_temp = df[df['Task'] == 'sent'& df['Langs']=='swa']
  labels=["Chanya","Wastani","Hasi"]
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for Sentiment Swahili:", res)

  labels = ["0","1","2"]
  df_temp = df[df['Task'] == 'xnli'& df['Langs']=='hau']
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for AfriXnli Hausa:", res)

  df_temp = df[df['Task'] == 'xnli'& df['Langs']=='swa']
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for AfriXnli Swahili:", res)

  df_temp = df[df['Task'] == 'mt'& df['Langs']=='swa']
  res = calculate_chrf(df_temp)
  scores.append(res['score'])

  avg_score = mean(int(n) for n in scores if n)
  print("Average performance score accross tasks and langs: ", avg_score)
  return avg_score
