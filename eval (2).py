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

PARAM_SIZE = 421939200.0
def zindi_score(submission_file):
  """
  This function generates the Lelapa AI Zindi score by combining model perfromance with model size
  This function relies on the structure of the file, so ensure no changes to the submission file structure
  are made

  Inputs: 
  submission_file: <string> Path to submisison file csv

  Outputs: <float> Lelapa AI Zindi score that appears on the leader board  
  """
  df = pd.read_csv(submission_file)
  avg_score = evaluate_zindi(df)
  size = df[df['Instruction']=='Model size']['Input Text'].astype(int)
  score = (avg_score + (1-(size/PARAM_SIZE))*avg_score)/2 #wat 
  return score

def calculate_chrf(df):
  """
  This function claculates CHRF when given a prediction and ground truth for translation

  Inputs: 
  df: <pandas dataframe> with prediction and ground truth for machine translation

  Outputs: 
  score: <dict> chrf results
  """
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

def process_likelihood(likelihood_str: str) -> List[float]:
    # clean the string to remove unwanted characters
    print(f"Raw Likelihood String: {likelihood_str}") 
    clean_str = likelihood_str.replace('tensor(', '').replace(')', '').strip()  # remove 'tensor(' and ')'
    clean_str = clean_str.replace('[[', '').replace(']]', '').strip()  # remove extra brackets
    clean_str = clean_str.replace(" device='cuda:0'", '').replace(" dtype=torch.float16", '').strip()  # remove device and dtype info
    clean_str = clean_str.replace('tensor', '').strip()  # remove any instances of 'tensor'
    
    # remove any empty strings caused by extra commas
    clean_str = clean_str.replace(',,', ',')  # remove duplicate commas if they exist
    
    # debugging output for verification
    print(f"Processed String: {clean_str}")  # debugging line
    
    # Convert to a list of floats
    likelihood = [float(x) for x in clean_str.split(',') if x.strip()]  # ensure non-empty strings are converted
    return likelihood



def loglikelihoods_to_predictions(log_likelihoods: List[float], labels: List[str]) -> List[str]:
    predictions = []

    for likelihoods in log_likelihoods:
      if isinstance(likelihoods,list):
        predicted_label_idx = np.argmax(likelihoods)
        predictions.append(labels[predicted_label_idx])
      else:
         print(f"Invalid likelihood format: {likelihoods}")
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
          # likelihood = [float(x) for x in row['Log-Likelihood'].strip('[]').split(',')]
          likelihood =process_likelihood(row['Log-Likelihood'])
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

  labels: <list> of target labels
  """
  log_likelihoods = df['Log-Likelihood'].astype(float)
  ground_truths = df['Targets']
  predictions = loglikelihoods_to_predictions(log_likelihoods, labels)
  accuracy, f1 = compute_f1_and_accuracy(predictions, ground_truths, labels)
  return f1

def evaluate_zindi(df):
  """
  This function is the same as main function above, it just does
  things in memory so the function can be used for the final eval
  as well.

  First step is to separate df such that each task has its own accuracy calc
  This turns the submissionn file into scores
  Inputs
  df: submission file df

  Outputs:
  Average score accross all the tasks
  """
  scores = []
  df_temp = df[(df['Task'] == 'sentiment') & (df['Langs'] == 'hausa')]
  labels = ["Kyakkyawa","Tsaka-tsaki","Korau"]
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for Sentiment Hausa:", res)

  df_temp = df[(df['Task'] == 'sentiment') & (df['Langs'] == 'swahili')]
  labels=["Chanya","Wastani","Hasi"]
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for Sentiment Swahili:", res)

  labels = ["0","1","2"]
  df_temp = df[(df['Task'] == 'xnli') & (df['Langs'] == 'hau')]  
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for AfriXnli Hausa:", res)

  df_temp = df[(df['Task'] == 'xnli') & (df['Langs']=='swa')]
  res = do_eval_compute(df_temp, labels)
  scores.append(res)
  print("Score for AfriXnli Swahili:", res)

  df_temp = df[df['Task'] == 'mmt']
  res = calculate_chrf(df_temp)
  print("Score for MMT:", res)
  scores.append(res['score']/100.0)

  avg_score = mean(float(n) for n in scores if not np.isnan(float(n)))
  print("Average performance score accross tasks and langs: ", avg_score)
  return avg_score
