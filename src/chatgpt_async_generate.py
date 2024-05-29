import numpy as np
from tqdm import tqdm
from langchain.llms import OpenAIChat, OpenAI
from langchain.chat_models import ChatOpenAI, openai
import json
import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio
import random
import re
import os

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

random.seed(9999)

parser = argparse.ArgumentParser()

parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--max_token", type=int, default=1000)
parser.add_argument("--augmentation", type=str, default='label_aug')
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--num_vote", type=int)
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()

with open(args.data_args_path, "r") as f:
    data_args = json.load(f)

data_args = data_args[:5]

with open(args.prompt, "r") as f:
    prompt = f.read()

# sort function
def dict_sort_by_id(item):
    parts = tuple(map(int, item['id'].split('-')))
    return parts


def most_frequent(data):
    return max(data, key=data.count)


# GPT input instance cache
all_model_inputs = []


# STEP2 type factuality verification
if args.type == 'infer_candidate':
    print("Start STEP2..")
    model_name = 'gpt-3.5-turbo-1106'

    ver_input_json = []
    not_searchable_instance = []
     
    for i, item in enumerate(data_args):
        in_KB = [cand for cand in list(item['knowledge'].keys()) if isinstance(item['knowledge'][cand], dict)]
        in_train = [cand for cand in list(item['knowledge'].keys()) if isinstance(item['knowledge'][cand], list)]
        cand_list = in_KB + in_train
 
        if cand_list == []: 
            not_searchable_dic = {}
            not_searchable_dic['id'] = item['id'] + '-' + str(1)  # k+1 == candidate index
            not_searchable_dic['context'] = item['context']
            not_searchable_dic['pred_entity'] = item['pred_entity']
            not_searchable_dic['pred_label'] = item['pred_label']
            not_searchable_dic['true_entity'] = item['true_entity']
            not_searchable_dic['true_label'] = item['true_label']
            not_searchable_dic['cand_list'] = None
            not_searchable_instance.append(not_searchable_dic)

        else:
            for k , cand in enumerate(cand_list):
                if cand in in_KB:
                    try:
                        definition = item['knowledge'][cand]['search_results']['d_t'][0]['definition']
                    except:  # not null
                        definition = None
                    try:
                        semantic_type = item['knowledge'][cand]['search_results']['d_t'][0]['semantic_type']
                    except:  # not null
                        semantic_type = None

                    if definition is None:
                        definition = 'not provided'
                    if semantic_type is None:
                        semantic_type = 'not provided'

                    verbalized_knowledge = f'The definition of {cand} is {definition}.\n The semantic type of {cand} is {semantic_type}.\n '

                    if (definition is None) & (semantic_type is None):
                        verbalized_knowledge = f'Although definition and semantic type of {cand} is not provided, {cand} is a valid biomedical entity.'

                elif cand in in_train:
                    if args.augmentation == 'train_entity_only':
                        verbalized_knowledge = f'Although definition and semantic type of {cand} is not provided, {cand} is a valid biomedical entity.'
                    else:
                        train_label = item['knowledge'][cand][-1]
                        verbalized_knowledge = f"The semantic type of {cand} is {train_label}.\n"

                # make new json
                verification_input = {}
                verification_input['id'] = item['id'] + '-' + str(k+1)  # k+1 == candidate index
                verification_input['context'] = item['context']
                verification_input['pred_entity'] = item['pred_entity']
                verification_input['pred_label'] = item['pred_label']
                verification_input['true_entity'] = item['true_entity']
                verification_input['true_label'] = item['true_label']
                verification_input['cand_list'] = cand_list
                verification_input['candidate'] = cand
                if cand in in_KB:
                    verification_input['source'] = 'KB'
                elif cand in in_train:
                    verification_input['source'] = 'trainset'
                else:
                    verification_input['source'] = 'KB'
                verification_input['verbalized_knowledge'] = verbalized_knowledge

                ver_input_json.append(verification_input)

    print("all instances", len(ver_input_json) + len(not_searchable_instance))
    print("verification input", len(ver_input_json) )
    print("not searhcable instances:",len(not_searchable_instance) )


    data_args = ver_input_json 

    if args.dataset_name == 'GENIA':
        for i, item in enumerate(data_args):
            all_model_inputs.append(prompt.format(context=item['context'],
                                                entity=item['candidate'],
                                                knowledge=item['verbalized_knowledge']))



elif args.type == 'STEP3_ablation_wo_rationale':
    print("Start STEP3 Ablation (wo rationale)..")
    model_name = 'gpt-3.5-turbo'

    not_searchable = []
    step3input = []


    for item in data_args:
        if item['per_cand_explanations'] == None:
            not_searchable.append(item)
        else:
            step3input.append(item)

    print("not_searchable -> none assign:", len(not_searchable))
    print("step3input:", len(step3input))
    print("whole instance:", len(not_searchable) + len(step3input))

    data_args = step3input

    for item in data_args: 
        context = item['context']
        step2output = item['per_cand_explanations']


        cand_exp_pairs = []
        for cand in step2output:
            pair = [cand['cand_exp_pair'][0], cand['cand_label']]
            cand_exp_pairs.append(pair)



        all_model_inputs.append(prompt.format(context= context,
                                                pair='\n'.join([f"{pair}" for pair in cand_exp_pairs])))


elif args.type == 'ablation_select_n_gen_knowledge':
    print("Start Ablation..")
    model_name = 'gpt-3.5-turbo'

    for i, item in enumerate(data_args):
        all_model_inputs.append(prompt.format(candidates= list(item['heu_list'].keys()),
                                              ))



# Contextual Relevance Verification
elif args.type == 'STEP3_self_consistency':
    print("Start STEP3..[SelfConsistency]")
    model_name = 'gpt-3.5-turbo-1106'

    not_searchable = []
    step3input = []

    for item in data_args:
        if item['per_cand_explanations'] == None:
            not_searchable.append(item)

        else:
            step3input.append(item)


    print("not_searchable -> none assign:", len(not_searchable))
    print("step3input:", len(step3input))
    print("whole instance:", len(not_searchable) + len(step3input))
    data_args = step3input
    print("sampled_instance:", len(data_args))

 

    for item in data_args: 
        context = item['context']
        step2output = item['per_cand_explanations']

        
        cand_exp_pairs = []
        for cand in step2output:
            pair = cand['cand_exp_pair']
            cand_exp_pairs.append(pair)



        all_model_inputs.append(prompt.format(context= context,
                                                pair='\n'.join([f"{pair}" for pair in cand_exp_pairs])))


### baseline llm-Revision ###
elif args.type == 'baseline_llm-verification':
    print("Start Ablation..")
    model_name = 'gpt-3.5-turbo'

    for item in data_args:
        before_ver_prediction = item['revised_context']
        all_model_inputs.append(prompt.format(context = before_ver_prediction))



# do not use
else:
    print("Something Went Wrong :( ")



print("len(all_model_inputs):",len(all_model_inputs))


collected_predictions = []
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

total_cost = 0
lock = asyncio.Lock()

async def async_generate(llm, model_input, i):
    global total_cost
    global collected_predictions
    system_message = SystemMessage(
        content="Follow the instruction to revise NER prediction using knowledge (definition and semantic type).")
    while True:
        try:
            response = await llm.agenerate([[system_message, HumanMessage(content=model_input)]])

            prompt_tokens = response.llm_output['token_usage']['prompt_tokens']
            completion_tokens = response.llm_output['token_usage']['completion_tokens']
            
            prompt_tokens_cost = prompt_tokens / 1000 * 0.001   
            completion_tokens_cost = completion_tokens / 1000 * 0.002

            print("completion_tokens:", completion_tokens)
            print("prompt_tokens_cost:", prompt_tokens_cost)
            print("completion_tokens_cost:", completion_tokens_cost)
            
            total_cost += (prompt_tokens_cost + completion_tokens_cost)

            print("total_cost:",total_cost)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None
    async with lock:

        cur_data = data_args[i]

        cur_data['prompt'] = model_input
        
        if args.type == 'STEP3_self_consistency':
            votings = [i.text for i in response.generations[0]]
            cur_data['revision'] = votings


        else:
            cur_data['revision'] = response.generations[0][0].text
        
        collected_predictions.append(cur_data)

        if len(collected_predictions) % 30 == 0:
            print(f"Expected Cost: {round(total_cost / len(collected_predictions) * len(data_args))}")


if args.type == 'STEP3_self_consistency':
    temperature = 0.9
    num_vote = args.num_vote
    print("num_vote:", num_vote)
else:
    temperature = 0
    num_vote = 1

async def generate_concurrently(all_model_input, model_name):
    llm = ChatOpenAI(temperature=temperature, n=num_vote, model_name=model_name)
    tasks = [async_generate(llm, model_input, i) for i, model_input in enumerate(all_model_input)]
    await tqdm_asyncio.gather(*tasks)


async def main():
    await generate_concurrently(all_model_inputs, model_name)


if __name__ == "__main__":

    asyncio.run(main())
    print("collected_predictions:",len(collected_predictions))



    if args.type == 'infer_candidate':
        for i, item in enumerate(data_args):
            candidate = item['candidate']
            explanation = item['revision']
            label = explanation.split()[-1].strip('.')
            item['cand_exp_pair'] = (candidate, explanation)
            item['candidate_label'] = label

        collected_predictions += not_searchable_instance
        collected_predictions_sorted = sorted(collected_predictions, key = dict_sort_by_id)


        grouped_dict = {}
        for item in collected_predictions_sorted:
            id_parts = item["id"].split("-")[:2] 
            group_key = "-".join(id_parts)

            if group_key not in list(grouped_dict.keys()):
                grouped_dict[group_key] = []


            grouped_dict[group_key].append(item)

        group_keys = list(grouped_dict.keys())
        print("group_keys_length:",len(group_keys))


        ent_level_instance = []
        for g_key in group_keys:

            ent_level_dic = {}
            ent_level_dic['id'] = g_key

            
            for cand_items in grouped_dict[g_key]:
                cand_level_instance = []
                
                if cand_items['cand_list'] == None: # non searchable instancss
                    ent_level_dic['context'] = cand_items['context']
                    ent_level_dic['pred_entity'] = cand_items['pred_entity']
                    ent_level_dic['pred_label'] = cand_items['pred_label']
                    ent_level_dic['true_entity'] = cand_items['true_entity']
                    ent_level_dic['true_label'] = cand_items['true_label']
                    cand_level_instance = None
                    continue


                else:
                    ent_level_dic['context'] =cand_items['context']
                    ent_level_dic['pred_entity'] = cand_items['pred_entity']
                    ent_level_dic['pred_label'] = cand_items['pred_label']
                    ent_level_dic['true_entity'] = cand_items['true_entity']
                    ent_level_dic['true_label'] = cand_items['true_label']
                    ent_level_dic['cand_list'] = cand_items['cand_list']
                    

                    # else:
                    cand_dic = {}
                    cand_dic['cand_id'] = cand_items['id']
                    cand_dic['cand'] = cand_items['candidate']
                    cand_dic['source'] = cand_items['source']
                    cand_dic['verbalized_knowledge'] = cand_items['verbalized_knowledge']
                    cand_dic['cand_exp_pair'] = cand_items['cand_exp_pair']
                    cand_dic['cand_label'] = cand_items['candidate_label']


                    cand_level_instance.append(cand_dic)

            ent_level_dic['per_cand_explanations'] = cand_level_instance
            ent_level_instance.append(ent_level_dic)
        print("ent_level_instance:", len(ent_level_instance))

        with open(args.save_path, "w") as f:
            json.dump(ent_level_instance, f, indent=4)



    elif (args.type == 'select_candidate') | (args.type == 'STEP3_ablation_wo_rationale') :

        parsing_error = 0
        for i, item in enumerate(data_args):

            revision = item['revision']
            if revision[-1] is None:
                print(f"{i}th instance : none verification")
                final_entity = revision[0]
                final_label = revision[1]
                item['final_answer'] = (final_entity, 'None')
            else:
                match = re.search(r"\('([^']*)'\, ([^)]*)\)\.", revision)

                if match:
                    final_entity = match.group(1)
                    final_label = match.group(2)
                    item['final_answer'] = (final_entity, final_label)

                else:
                    match = re.search(r'\(([^)]+), ([^)]+)\)', revision)
                    if match:
                        final_entity = match.group(1)
                        final_label = match.group(2)
                        item['final_answer'] = (final_entity, 'None')
                    else:
                        parsing_error += 1
                        item['final_answer'] = 'parsing_error'


        for i, item in enumerate(not_searchable):
            item['final_answer'] = (None, 'None')   

        
        collected_predictions += not_searchable
        collected_predictions_sorted = sorted(collected_predictions, key=dict_sort_by_id)
        print("collected_predictions_sorted(verify + none):", len(collected_predictions_sorted))

        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)



    elif args.type == 'ablation_select_n_gen_knowledge':
        
        for i, item in enumerate(data_args):
            item['per_cand_explanations'] = []
            revision = item['revision']
            pattern = r'\[entity:.*?\]'
            matches = re.findall(pattern, revision)
            if matches:
                for i, match in enumerate(matches):
                    
                    entity_pattern = r"entity: '(.*?)'"
                    label_pattern = r"label: '(.*?)'"
                    explanation_pattern = r"explanation: '(.*?)'"
                    
                    entity_match = re.search(entity_pattern, match)
                    label_match = re.search(label_pattern, match)
                    explanation_match = re.search(explanation_pattern, match)
                    
                    if entity_match and label_match and explanation_match:
                        entity = entity_match.group(1)
                        label = label_match.group(1)
                        explanation = explanation_match.group(1)
                        item['per_cand_explanations'].append({'cand_exp_pair': [entity,explanation]})
                    else:
                        continue

        collected_predictions_sorted = sorted(collected_predictions, key = dict_sort_by_id)
        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)



    elif args.type == 'STEP3_self_consistency':
        
        parsing_error = 0
        for i, item in enumerate(data_args):
            item['votes'] = []
            for revision in item['revision']:
                if revision[-1] is None:
                    print(f"{i}th instance : none verification")
                    final_entity = revision[0]
                    final_label = revision[1]
                    item['votes'].append((final_entity, 'None'))
                else:
                    match = re.search(r"\('([^']*)'\, ([^)]*)\)\.", revision)

                    if match:
                        final_entity = match.group(1)
                        final_label = match.group(2)
                        item['votes'].append((final_entity, final_label))

                    else:
                        match = re.search(r'\(([^)]+), ([^)]+)\)', revision)
                        if match:
                            final_entity = match.group(1)
                            final_label = match.group(2)
                            item['votes'].append((final_entity, 'None'))
                        else:
                            parsing_error += 1
                            item['votes'].append('parsing_error')
            item['final_answer'] = most_frequent(item['votes'])

        for i, item in enumerate(not_searchable):
            item['final_answer'] = (None, 'None')   

        collected_predictions += not_searchable
        collected_predictions_sorted = sorted(collected_predictions, key=dict_sort_by_id)
        print("collected_predictions_sorted(verify + none):", len(collected_predictions_sorted))

        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)


    else:
        collected_predictions_sorted = sorted(collected_predictions, key = dict_sort_by_id)
        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)