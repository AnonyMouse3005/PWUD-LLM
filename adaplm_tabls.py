import torch
import numpy as np
import tqdm
from numpy.random import choice
import pandas as pd
import torch
from multiprocessing.pool import ThreadPool as Pool
import os
import json
import copy
import itertools
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from pathlib import Path
import yaml
from datetime import datetime
import pytz
import ast

from dotenv import load_dotenv
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random,
)

from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from numpy.random import choice

# ----------------------------------------------------------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------------------------------------------------------
class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, records, true_labels, separator = '#',
                 train = True, seed = 0, force_balance=False):
        '''
            param records: table as text. list of table rows where text has the separator in-between (if labeled).
            param true_labels: actual GT label of this example (empty list means records are unlabeled).
            param force_balance: makes the dataset very balanced (oversampling minority class).
        '''
        self.seed = seed
        self.dataset_name = dataset_name

        assert len(records) > 0

        data = []
        if true_labels:  # labeled records
            for r in records:
                d, label_text = r.split(separator)
                data.append((d, label_text.strip()))

            train_data, test_data, train_labels, test_labels \
                = train_test_split(data, true_labels, test_size = 0.4,
                                random_state = seed, stratify = true_labels)

            if train:
                self.data = []
                self.labels = []

                if force_balance:
                    # obtain class wise counts.
                    cls_counts = defaultdict(int)
                    cls_indices = defaultdict(list)

                    for i, cls in enumerate(train_labels):
                        cls_indices[cls].append(i)
                        cls_counts[cls] +=1

                    # class with max samples.
                    max_length = max(cls_counts.values())

                    for cls in cls_indices:
                        np.random.seed(seed)
                        indices = choice(cls_indices[cls], size = max_length, replace = True)
                        self.data.extend(subset(train_data, indices))
                        self.labels.extend(subset(train_labels, indices))

                else:
                    self.data = train_data
                    self.labels = train_labels

            # test data
            else:
                self.data = test_data
                self.labels = test_labels

        else:  # unlabeled records
            for r in records:
                data.append((r, None))
            self.data = data
            self.labels = []

        # separate out the data and label texts. - comes handy for test() that seek only data text.
        self.data_texts = []
        self.label_texts = []
        for d, label_text in self.data:
            self.data_texts.append(d)
            self.label_texts.append(label_text)

        self.length = len(self.data)

    def __getitem__(self, item):
        '''
        returns a pair of (data text, label) for the given index.
        '''
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.length

    def pseudo_label(self, labels):
        self.labels = labels


def merge_datasets(datasets: list) -> pd.DataFrame:
    data_texts, labels = [], []
    for dataset in datasets:
        for data_text, label in zip(dataset.data_texts, dataset.labels):
            data_texts.append(data_text)
            labels.append(label)

    return pd.DataFrame.from_dict({'text': data_texts, 'label': labels})


def sanity_check(model, subdata_desc, prefix, suffix, questions):
    '''
    returns '1' if answer is consistent with the original features, '0' otherwise;
    also ensure each subdata description is not too long
    '''
    prompt = f'''{prefix}

{subdata_desc}

{suffix}
{questions}'''

    response = openai.completions.create(
        model = model,
        prompt = prompt,
        temperature = 1
    )
    answer = response.choices[0].text.strip()
    print('total number of words: ', len(subdata_desc.split()))
    print('answer:\n', answer)
    decision = input("pass or not ('1' for yes, '0' for no)? ")
    return decision

GROUPS = {
    'SC': "community and social support",
    'DA': "drug accessibility and use patterns",
    'OD': "drug overdose",
    'TX': "substance use treatment",
    'AC': "adverse childhood experiences",
    'CJ': "criminal justice involvement",
    'TB': "tobacco use",
    'AL': "alcohol use",
    'ID': "injection drug use",
    'ND': "non-injection drug use",
    'DM': "demographics"
}
def data_conversion(df, params):
    texts = []
    for _, row in df.iterrows():
        # Part i
        subdata_descs = ""
        for group, group_desc in GROUPS.items():
            group_data = row[f'data_{group}']
            if pd.notnull(group_data):
                group_data = ast.literal_eval(group_data)
                prefix_g_prompt = params['prefix_g_prompt'].replace('<GROUP>', group_desc)
                prefix_qa_prompt = params['prefix_qa_prompt'].replace('<GROUP>', group_desc)
                subdata_conv_prompt = f'''{prefix_g_prompt}

{group_data}

{params['suffix_prompt']}.'''

                decision = '0'
                while not int(decision):
                    response = openai.completions.create(
                        model = params['model'],
                        prompt = subdata_conv_prompt,
                        temperature = 1,
                        max_tokens = 250
                    )
                    subdata_desc = response.choices[0].text.strip()
                    decision = sanity_check(params['model'], subdata_desc, prefix_qa_prompt,
                                            params['suffix_qa_prompt'], '\n'.join(group_data.keys()))

                if subdata_descs:
                    subdata_descs += '\n' + subdata_desc
                else:
                    subdata_descs += subdata_desc

        # Part ii
        data_conv_prompt = f'''{params['meta_data']}

{params['prefix_prompt']}
{subdata_descs}

{params['suffix_prompt']}.'''

        final_decision = '0'
        while not int(final_decision):
            response = openai.completions.create(
                model = params['model'],
                prompt = data_conv_prompt,
                temperature = 1,
                max_tokens = 500
            )
            data_desc = response.choices[0].text.strip()
            print('data description:\n', data_desc)
            print('total number of words: ', len(data_desc.split()))
            final_decision = input("pass or not ('1' for yes, '0' for no)? ")
        texts.append(data_desc)
    
    df['text'] = texts
    return df


# ----------------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------------
class Summary(torch.nn.Module):
    def __init__(self, meta_data = "", inference_prompt = "", cot_params = None,
                 temperature = None, model = None, classes = None, suffix = None, system_message = None):
        super(Summary, self).__init__()
        self.meta_data = meta_data
        self.inference_prompt = inference_prompt
        self.temperature = temperature
        self.model = model  # PLM API
        self.classes = classes
        self.summary_prompt = None
        self.system_message = system_message
        self.summary = None
        self.cot_params = cot_params
        self.suffix = suffix

    def forward(self, inps):
        '''
            inps: list of data texts.
            returns : predicted classes.
        '''
        assert self.summary != None

        def predict_example(inp):
            if type(self.temperature).__name__ == 'list':
                temperature = np.random.uniform(low = self.temperature[0], high = self.temperature[1])
            else:
                temperature = self.temperature

            # prompt for first-stage prediction.
            prompt = f'''Knowledge to Prediction.

{self.meta_data}

{self.summary}

------------------------------------------------------
Now {inp[0].lower() + inp[1:]}.

{self.inference_prompt if self.cot_params is None else self.cot_params[0]}.'''
            inp_pred, top_logprobs = get_text_completion(model = self.model,
                                                         prompt = prompt,
                                                         temperature = 0 if self.cot_params is None else temperature,
                                                         max_tokens = 200,
                                                         suffix = self.suffix if self.cot_params is None else None)

            # prompt for second-stage prediction.
            if self.cot_params is not None:
                prompt = f'''{prompt}

{inp_pred}

{self.cot_params[1]}'''
                inp_pred, top_logprobs = get_text_completion(model = self.model,
                                                             prompt = prompt,
                                                             temperature = 0,
                                                             max_tokens = 200,
                                                             suffix = self.suffix)

            inp_pred = inp_pred.lower()
            # print(inp_pred)

            assigned = False
            for cls in self.classes:
                if cls in inp_pred:
                    inp_pred = cls
                    assigned = True
                    break

            if not assigned:
                inp_pred = choice(self.classes)

            return inp_pred, top_logprobs

        #--------------------------------------------------
        with Pool(10) as pool:
            outputs = list(tqdm.tqdm(pool.imap(predict_example, inps), total=len(inps)))

        inp_preds = [i[0] for i in outputs]
        top_logprobs = [i[1] for i in outputs]
        return inp_preds, top_logprobs


class SummaryBoosting(torch.nn.Module):
    def __init__(self, classes):
        super(SummaryBoosting, self).__init__()
        self.summaries = {}
        self.alpha = {}
        self.label_mapping = {}
        self.classes = classes
        self.best_round = None

    def forward(self, inps):

        assert self.best_round != None

        round_preds = {}
        for t in range(0, self.best_round + 1):
            print('round ', t)
            # get preds from each weak learner.
            preds, _ = self.summaries[t](inps)

            # apply label mapping
            learner_preds = []
            for pred in preds:
                learner_preds.append(self.label_mapping[t][pred])

            round_preds[t] = learner_preds

        _, inp_preds = compute_combined_error(round_preds, self.alpha, self.classes, true_labels=None)

        return inp_preds, [None] * len(inp_preds)

# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------

MODELS = {'instruct': 'gpt-3.5-turbo-instruct', 'scibert': 'scibert_scivocab_uncased', 'roberta': 'roberta-base'}

@retry(wait=wait_random(min=1, max= 120), stop=stop_after_attempt(10))
def get_text_completion(model, prompt, temperature, max_tokens, suffix = None):
    response = openai.completions.create(
        model = model,  # ,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        logprobs=1,
        suffix = suffix,
        # top_k=50,
        # top_p = 0.95
        # stop = '\n'
    )
    inp_pred = response.choices[0].text.strip()
    # top_logprobs = [dict(d) for d in response.choices[0]['logprobs']['top_logprobs']]
    top_logprobs = response.choices[0].logprobs.top_logprobs

    return inp_pred, top_logprobs

def safe_mkdir(path, force_clean=False):
    if os.path.exists(path) and force_clean:
        os.rmdir(path)
    os.makedirs(path, exist_ok=True)
    return

class Parameters:
    model_path = "models"
    datasets_path = 'datasets'

# argmax of the weighted class score for each example.
def compute_combined_error(roundwise_preds, alpha, classes, true_labels = None):
    combined_preds = []
    for i in range(len(roundwise_preds[0])):
        cls_score = {}
        for cls in classes:
            score = 0.
            for t in range(len(roundwise_preds)):
                score += alpha[t] * (roundwise_preds[t][i] == cls)
            cls_score[cls] = score
        # take argmax of the cls with highest score.
        inp_pred = classes[np.argmax([cls_score[cls] for cls in classes])]
        # final prediction of the example.
        combined_preds.append(inp_pred)

    if true_labels != None:

        error_rate = 0

        for i in range(len(combined_preds)):
            incorrect = combined_preds[i] != true_labels[i]
            error_rate += int(incorrect)

        error_rate /= len(true_labels)

        return round(error_rate, 3), combined_preds

    return None, combined_preds

def subset(elements, indices):
    '''
    selects a subset of elements
    '''
    selected_elements = []
    for i in indices:
        selected_elements.append(elements[i])
    return selected_elements

def Subset(dataset, indices):
    '''
    my own version of the pytorch Subset but can access the class members.
    phew.. only if the Subset inherited the class members and methods !!
    '''
    dataset = copy.deepcopy(dataset)
    dataset.data = subset(dataset.data, indices)
    dataset.data_texts = subset(dataset.data_texts, indices)
    dataset.label_texts = subset(dataset.label_texts, indices)
    dataset.labels = subset(dataset.labels, indices)
    dataset.length = len(indices)
    return dataset

def get_error_rate(model, inps, true_labels):
    """
    :param
        inps: input prompt.
        true labels: actual labels.
        classes: target classes for the dataset.
    """
    error_rate = 0

    inp_preds, top_logprobs = model(inps)

    for i in range(len(true_labels)):
        incorrect = inp_preds[i] != true_labels[i]
        error_rate += int(incorrect)

    error_rate /= len(true_labels)

    if type(inp_preds).__name__ != 'list':
        inp_preds = list(inp_preds.astype(float))

    return round(error_rate, 3), inp_preds, top_logprobs

def normalize(elements):
    tot = sum(elements)
    new_elements = []
    for e in elements:
        new_elements.append(e / tot)
    return new_elements

def random_sample(dataset, w, num_examples = 20):
    indices = list(range(len(dataset)))
    sampled_indices = choice(indices, size = num_examples, replace = True, p = w)
    return Subset(dataset, sampled_indices)

def cluster_sample(embeddings, dataset, w = None, num_examples = 20):
    selected_indices = []

    # obtain class wise counts.
    cls_counts = defaultdict(int)
    cls_indices = defaultdict(list)

    data_texts, labels = dataset.data_texts, dataset.labels

    for i, cls in enumerate(labels):
        cls_indices[cls].append(i)
        cls_counts[cls] += 1

    # do sampling per each class.
    for cls, indices in cls_indices.items():
        # samples needed for this class.
        expected_count = int(cls_counts[cls]/len(labels) * num_examples)

        cls_embeddings = embeddings[indices]

        clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine',
                                             linkage='average', distance_threshold=0.05)
        clustering.fit(cls_embeddings)
        cluster_assignment = clustering.labels_

        # compute weights of each cluster
        cluster_counts = defaultdict(int)
        cluster_indices = defaultdict(list)
        for i, cluster in enumerate(cluster_assignment):
            cluster_counts[cluster]+=1
            cluster_indices[cluster].append(i)

        cluster_weights = defaultdict(float)
        for cluster, count in cluster_counts.items():
            cluster_weights[cluster] = len(labels)/count

        # map to samples.
        sample_weights = []
        for cluster in cluster_assignment:
            sample_weights.append(cluster_weights[cluster])

        # normalize weights.
        sample_weights = normalize(sample_weights)

        if w is not None:
            for i in range(len(indices)):
                w_cls = w[indices[i]]
                sample_weights[i] *= w_cls
            sample_weights = normalize(sample_weights)

        sampled_indices = choice(indices, size = expected_count, replace = False, p = sample_weights)
        selected_indices.extend(sampled_indices)

    if len(selected_indices) < num_examples:
        # if the number of examples is less than the required, sample the remaining from the entire dataset.
        indices = list(range(len(dataset)))
        remaining = num_examples - len(selected_indices)
        sampled_indices = choice(indices, size = remaining, replace = False)
        selected_indices.extend(sampled_indices)

    return Subset(dataset, selected_indices)

def get_gpt3_embeddings(examples, model_name = "text-embedding-ada-002"):
    '''
        examples: list of (data text, label text) pairs
        model_name: target GPT3 embedding model
        returns: embeddings matrix of size len(examples) x embedding-dims of chosen model.
    '''
    texts = [d for d, l in examples]
    embeddings_output = openai.embeddings.create(input=texts, model=model_name)

    embeddings = []
    for e in embeddings_output['data']:
        embeddings.append(e['embedding'])

    embeddings = np.array(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

# ----------------------------------------------------------------------------------------------------------------------
# Training code
# ----------------------------------------------------------------------------------------------------------------------
def train_summary_boosting(base_learner, dataset_name, records, true_labels, separator,
                  summarize_prompt, seed = 0,
                  sampling = 'cluster',
                  num_examples = 32,
                  num_rounds = 50, save_suffix="_summary_boosting", reuse = False):
    train_size = 5 / 6

    model = SummaryBoosting(classes=base_learner.classes)

    checkpoint_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix,
                                   dataset_name)
    final_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                              f"final_chkpt_seed{seed}.json")
    safe_mkdir(checkpoint_path)

    whole_train_dataset = TabularDataset(dataset_name, records[0], true_labels, separator, train = True, force_balance = True)
    train_size_int = int(len(whole_train_dataset) * train_size)
    rand_indices = np.arange(len(whole_train_dataset))
    np.random.shuffle(rand_indices)
    train_indices = rand_indices[:train_size_int]
    val_indices = rand_indices[train_size_int:]
    train_dataset = Subset(whole_train_dataset, train_indices)
    val_dataset = Subset(whole_train_dataset, val_indices)
    test_dataset = TabularDataset(dataset_name, records[0], true_labels, separator, train = False)

    best_val_err = np.inf
    best_round = -1
    train_preds = {}
    val_preds = {}
    patience = 0

    # init data weight distribution.
    w = [1/len(train_dataset)] * len(train_dataset)

    # resume training if the checkpoint exists.
    if Path(final_path).exists():
        with open(final_path, 'r') as f:
            stats = json.load(f)
            key_transform = lambda x : int(x) if x.isdigit() else x
            stats = {key_transform(k): v for k, v in stats.items()}
            # make all the string indices of int type.
            train_preds = {int(k): v for k, v in stats['train_preds'].items()}
            val_preds = {int(k): v for k, v in stats['val_preds'].items()}
            model.alpha = {int(k): v for k, v in stats['alpha'].items()}
            model.label_mapping = {int(k): v for k, v in stats['label_mapping'].items()}
            for r, summary in stats['summaries'].items():
                new_learner = copy.deepcopy(base_learner)
                new_learner.summary = summary
                model.summaries[int(r)] = new_learner

        keys = list(stats.keys())
        key = -2 if keys[-1] == 'best' else -1
        last_round = int(keys[key])
        w = stats[last_round]['w']

        for i in range(0, last_round + 1):
            combined_val_err_at_round_i = stats[i]['combined_val_err']
            if combined_val_err_at_round_i <= best_val_err:
                best_val_err = combined_val_err_at_round_i
                best_round = i

    else:
        stats = {
            'seed': seed,
            'summarize_prompt': summarize_prompt,
            'base_learner_params': {k: v for k, v in vars(base_learner).items() if not k.startswith('_')}
        }
        del stats['base_learner_params']['summary']
        last_round = -1

    embeddings = get_gpt3_embeddings(examples=train_dataset.data)

    if len(model.classes) == 2:
        stopping_threshold = 0.42
    if len(model.classes) == 3:
        stopping_threshold = 0.55
    if len(model.classes) == 4:
        stopping_threshold = 0.65

    # go through the rounds.
    for r in range(last_round+1, num_rounds):
        # obtain a weak learner for the current round.
        h = copy.deepcopy(base_learner)

        print('calculating train error.')
        result = get_weak_learner(h, train_dataset,
                            embeddings, w, summarize_prompt,
                            num_times=20, num_examples = num_examples,
                            sampling = sampling,
                            stopping_threshold = stopping_threshold)

        # if result is none, then need to set the hyperparameters properly.
        if result is None:
            break

        # assert result != None

        model.summaries[r], model.label_mapping[r], \
            train_err, train_preds[r], misclassifications = result

        print('calculating val error.')
        val_err, val_preds[r], _ = get_error_rate(model.summaries[r], val_dataset.data_texts, val_dataset.labels)

        # compute alpha
        model.alpha[r] = np.log((1 - train_err) / train_err) + np.log(len(model.classes) - 1)

        # reweigh w.
        new_w = []
        for i in range(len(misclassifications)):
            incorrect = misclassifications[i]
            new_w_i = w[i] * np.exp(model.alpha[r] * incorrect) # / sum(w)
            new_w.append(new_w_i)
        new_w = normalize(new_w)
        w = new_w

        # compute combined train error until this point.
        combined_train_err, combined_train_preds = compute_combined_error(train_preds,
                                                                          model.alpha,
                                                                          model.classes, train_dataset.labels)

        # compute combined val error until this point.
        combined_val_err, combined_val_preds = compute_combined_error(val_preds,
                                                                      model.alpha,
                                                                      model.classes, val_dataset.labels)
        # compare val error.
        if combined_val_err <= best_val_err:
            best_val_err = val_err
            best_round = r
            patience = 0
        else:
            patience += 1

        stats['train_preds'] = train_preds
        stats['val_preds'] = val_preds
        stats['alpha'] = model.alpha
        stats['label_mapping'] = model.label_mapping
        stats['summaries'] = {i: model.summaries[i].summary for i in range(len(model.summaries)) }

        stats[r] = {'train_err': train_err,
                    'misclassifications': misclassifications,
                    'val_err': val_err,
                    'w': w,
                    'combined_train_err': combined_train_err, 'combined_train_preds': combined_train_preds,
                    'combined_val_err': combined_val_err, 'combined_val_preds': combined_val_preds,}

        print(f'round {r + 1}/{num_rounds}, train err = {train_err:.4f}, '
              f'combined train err = {combined_train_err:.4f}, combined val err = {combined_val_err:.4f}')
        print('-' * 50)

        json.dump(stats, open(final_path, 'w'), indent=4)

        if combined_train_err < 0.05 or patience == 10:
            break

    model.best_round = best_round

    if reuse:  # leverage SummaryBoost as part of the AdaPLM-TabLS framework
        unlabeled_dataset = TabularDataset(dataset_name, records[1], [])
        inp_preds, _ = model(unlabeled_dataset.data_texts)  # provide pseudo-labels to the unlabeled dataset
        unlabeled_dataset.pseudo_label(inp_preds)
        augmented_dataset = merge_datasets([whole_train_dataset, unlabeled_dataset])
        now = datetime.now(pytz.utc).strftime("%Y%m%d-%H%M%S")
        augmented_filepath = os.path.join(Parameters.datasets_path, dataset_name, f"{dataset_file.split('.')[0]}_augmented_{now}.csv")
        # augmented_dataset.to_csv(augmented_filepath, index=False)
        return augmented_dataset
    else:
        print('calculating test error.')

        test_err, test_preds, test_logprobs = get_error_rate(model, test_dataset.data_texts, true_labels=test_dataset.labels)

        stats['best'] = {'round': best_round, 'combined_train_err': stats[best_round]['combined_train_err'],
                        'combined_val_err': stats[best_round]['combined_val_err'],
                        'train_err': stats[best_round]['train_err'],
                        'val_err': stats[best_round]['val_err'],
                        'test_err': test_err,
                        'test_preds': test_preds}

        print(f"test err = {test_err:.4f}, combined val err = {stats[best_round]['combined_val_err']:.4f}, "
            f"combined train err = {stats[best_round]['combined_train_err']:.4f}")

        json.dump(stats, open(final_path, 'w'), indent=4)

        return test_err, test_preds, test_logprobs


def get_weak_learner(model, train_dataset, embeddings, w, summarize_prompt,
                       num_times = 25, num_examples = 32, sampling = 'cluster', stopping_threshold = 1.00):

    label_mappings = []
    for combo in itertools.permutations(model.classes):
        l_map = dict(list(zip(combo, model.classes)))
        label_mappings.append(l_map)

    for t_ in range(num_times):  # resample for generating a good summary
        num_examples = min(num_examples, len(embeddings))
        if sampling == 'cluster':
            sampled_dataset = cluster_sample(embeddings, train_dataset, w = w, num_examples = num_examples)
        elif sampling == 'random':
            sampled_dataset = random_sample(train_dataset, w=w, num_examples=num_examples)

        examples = ""
        for i, ((d_text, l_text), _) in enumerate(sampled_dataset):
            examples += f"{i+1}. {d_text} ### {l_text}\n"

        prompt = f'''{model.meta_data}

{examples}

{summarize_prompt}.'''

        response = openai.completions.create(
            model = model.model,
            prompt = prompt,
            temperature = model.temperature,
            max_tokens = 500,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        summary = response.choices[0].text.strip(':').strip()
        print('summary: ', summary)

        model.summary = summary

        train_err, train_preds, _ = get_error_rate(model, train_dataset.data_texts,
                                                true_labels= train_dataset.labels)

        print(f'time = {t_+1}/{num_times}, unweighted train err = {train_err}')

        for l_map in label_mappings:
            # compute weighted train error.
            weighted_train_err = 0.

            inp_preds = []
            misclassifications = []
            for i in range(len(train_dataset)):
                pred = l_map[train_preds[i]]
                gt = train_dataset.labels[i]
                incorrect = 1. * (pred != gt)
                weighted_train_err += w[i] * incorrect
                inp_preds.append(pred)
                misclassifications.append(incorrect)

            weighted_train_err /= sum(w)

            print('weighted train err: ', weighted_train_err)

            if weighted_train_err <= stopping_threshold:
                return model, l_map, weighted_train_err, inp_preds, misclassifications

    # if the algo didnt produce a weak learner in given number of times.
    return None

def load_summary_boosting(base_learner, dataset_name, seed, save_suffix):
    model = SummaryBoosting(classes= base_learner.classes)

    final_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                              f"final_chkpt_seed{seed}.json")

    if Path(final_path).exists():
        best_val_err = np.inf
        with open(final_path, 'r') as f:
            stats = json.load(f)
            key_transform = lambda x : int(x) if x.isdigit() else x
            stats = {key_transform(k): v for k, v in stats.items()}
            model.alpha = {int(k): v for k, v in stats['alpha'].items()}
            model.label_mapping = {int(k): v for k, v in stats['label_mapping'].items()}
            for r, summary in stats['summaries'].items():
                new_learner = copy.deepcopy(base_learner)
                new_learner.summary = summary
                model.summaries[int(r)] = new_learner

        keys = list(stats.keys())
        key = -2 if keys[-1] == 'best' else -1
        last_round = int(keys[key])

        for i in range(0, last_round + 1):
            combined_val_err_at_round_i = stats[i]['combined_val_err']
            if combined_val_err_at_round_i <= best_val_err:
                best_val_err = combined_val_err_at_round_i
                model.best_round = i

        return model

    else:

        return None



# ----------------------------------------------------------------------------------------------------------------------
# Orchestrate Experiments
# ----------------------------------------------------------------------------------------------------------------------
def experiment_s_boosting(base_learner, dataset_name, records: list, true_labels: list, separator,
                 summarize_prompt,
                 seed,
                 num_rounds = 25,
                 experiment_name="instruct_prefix_tldr",
                 sampling = 'cluster',
                 num_examples = 32,
                 train = True):
    '''
    params:
        model: one of the MODELS
        dataset_name: caesarian/iris like that.
        records: [list of text containing both data and label sep by "###", list of text containing (unlabeled) data]
        seed: 0, 7, 16 like that.
        summarize_prompt: string such as ""
        num_epochs: 40
        experiment_name: use to refer to the experiment - to save while training or load back after training.
        train: if True then trains and dumps or just loads back in and computes test err.
    '''
    if train:
        err, preds, _ = train_summary_boosting(base_learner, dataset_name, records, true_labels, separator,
                                               summarize_prompt,
                                               seed,
                                               num_rounds = num_rounds,
                                               sampling = sampling,
                                               num_examples = num_examples,
                                               save_suffix=experiment_name)
    else:
        summary_model = load_summary_boosting(base_learner, dataset_name, seed, save_suffix = experiment_name)

        if summary_model is None:
            print('first train and save model, then call test.')

        dataset = TabularDataset(dataset_name, records[0], true_labels, separator = separator, train=False)
        err, preds, _ = get_error_rate(summary_model, dataset.data_texts, dataset.labels)
    return err


TIME_RANGES = {
    'A': "6 to 8 months",
    'B1': "8 to 10 months",
    'B2': "10 to 14 months",
    'C': "6 to 8 months"
}
def run_experiment_s_boosting(dataset_name, yml_file, train = True, problem = 'A', drug = 'meth', conversion = False):
    with open(os.path.join(Parameters.datasets_path, dataset_name, yml_file), "r") as fin:
        yml_filepath = os.path.join(Parameters.datasets_path, dataset_name, f"{yml_file.split('.')[0]}_{problem}_{drug}.yml")
        with open(yml_filepath, "w") as fout:
            for line in fin:
                fout.write(line.replace('<PROBLEM>', problem))
                fout.write(line.replace('<DRUG>', drug))
                fout.write(line.replace('<TIME_RANGE>', TIME_RANGES[problem]))

    global dataset_file
    with open(yml_filepath, 'r') as stream:
        params = yaml.safe_load(stream)
        summarize_prompt = params.pop('summarize_prompt') if 'summarize_prompt' in params else None
        seed = params.pop('seed')
        separator = params.pop('separator')
        dataset_file = params.pop('dataset_file')
        num_epochs = params.pop('num_epochs')
        sampling = params.pop('sampling') if 'sampling' in params else 'cluster'
        num_examples = params.pop('num_examples') if 'num_examples' in params else 40

    os.remove(yml_filepath)

    params['model'] = MODELS[params['model']]

    file = os.path.join(Parameters.datasets_path, dataset_name, dataset_file)
    df = pd.read_csv(file)
    if 'text' not in df.columns:
        df = data_conversion(df, params)
    ids = df.index[df['label'].notnull()].to_list()
    records, true_labels = df.loc[ids, 'text'], list(df.iloc[ids, -2].astype(str)) # list(df['normal or abnormal'])
    unlabeled_records = df.loc[df.index[df['label'].isnull()], 'text']

    base_learner = Summary(**params)

    test_err = experiment_s_boosting(base_learner, dataset_name, [records, unlabeled_records], true_labels, separator,
                            summarize_prompt,
                            seed = seed,
                            num_rounds = num_epochs,
                            experiment_name="_" + yml_file.split('.')[0],
                            sampling = sampling,
                            num_examples = num_examples,
                            train = train)
    print('test err :', test_err)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Summary Boosting',
                    description='learns to summarize different subsets of data using GPT3 and combines them using boosting.')
    parser.add_argument('--dataset_name', type=str, default = 'PWUD', help='name of the dataset')
    parser.add_argument('--yml_file', type=str, default = 'pwud_prefix.yml', help='name of the yml file')
    parser.add_argument('--problem', type=str, default = 'A', help='label of the problem')
    parser.add_argument('--drug', type=str, default = 'meth', help='name of the drug to predict')
    parser.add_argument('--test', action="store_true", help='test instead of train.')
    parser.add_argument('--conversion', action="store_true", help='do data conversion beforehand.')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    yml_file = args.yml_file
    problem = args.problem
    drug = args.drug
    test = args.test
    conversion = args.conversion

    run_experiment_s_boosting(dataset_name, yml_file, train = not test, problem = problem, drug = drug, conversion = conversion)
