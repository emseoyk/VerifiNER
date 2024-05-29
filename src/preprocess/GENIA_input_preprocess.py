import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--FN_save_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True) # biobert / conner/ gptner
args = parser.parse_args()



with open(args.data_args_path, "r") as f:
    bio_input = json.load(f)

with open("data/GENIA/BIO_input/GENIA_test_BIO_final.json", "r") as f:
    genia_test_bio = json.load(f)


print("before preprocess:")

print(bio_input[0])
print("num_document:", len(bio_input))
print()


def extract_heuristic_cand(item, k=2):
    words = item['context'].split(' ')
    start = item['pred_span']['start']
    end = item['pred_span']['end']
    axis_words = words[start:end + 1]

    whole_combinations = []
    for w in range(end - start + 1):
        axis_word = axis_words[w]

        start_idx = max(0, start + w - k)
        end_idx = min(len(words), start + w + k + 1)

        sub_context = ' '.join(words[start_idx:end_idx])

        sub_words = sub_context.split()

        # CNN-like filtering
        for f in range(1, len(sub_words) + 1):
            filter_size = f

            filter_combinations = []
            for j in range(len(sub_words) - filter_size + 1):
                combination = sub_words[j:j + filter_size]
                offset = [start_idx + j, start_idx + (j + filter_size) - 1]

                if axis_word in combination:
                    combination = ' '.join(combination)
                    filter_combinations.append({combination: offset})


            whole_combinations.append(filter_combinations)

    unique_combs = []
    for c in whole_combinations:
        for ent in c:
            if ent not in unique_combs:
                unique_combs.append(ent)

    return unique_combs


def preprocess_bio_to_ver_input(test_BIO, type):
    # make new_dic with gold bio
    if type == 'conner' :
        print("ConNER")
        new_bio_input = []
        for gpt in test_BIO:
            new_dic = {}
            new_dic['doc_id'] = gpt['doc_id']
            new_dic['context'] = gpt['context']
            new_dic['gold_BIO'] = gpt['gold_BIO']
            new_dic['pred_BIO'] = gpt['pred_BIO']
            new_bio_input.append(new_dic)

    elif type == 'biobert':
        print("BioBERT")
        new_bio_input = []
        for gpt in test_BIO:
            new_dic = {}
            new_dic['doc_id'] = gpt['doc_id']
            new_dic['context'] = gpt['context']
            new_dic['gold_BIO'] = gpt['gold_BIO']
            new_dic['pred_BIO'] = gpt['pred_BIO']
            new_bio_input.append(new_dic)

    else: # gptner
        print("GPTNER")
        new_bio_input = []
        for gpt in test_BIO:
            new_dic = {}
            new_dic['doc_id'] = gpt['doc_id']
            new_dic['context'] = gpt['context']
            new_dic['gold_BIO'] = gpt['gold_BIO']
            new_dic['pred_BIO'] = gpt['gpt_BIO']
            new_bio_input.append(new_dic)



    # make gold labels and pred labels
    for k, item in enumerate(new_bio_input):
        str_words = item['context'].split(' ')
        gold_BIO = item['gold_BIO']
        pred_BIO = item['pred_BIO']

        item['gold_labels'] = []
        item['pred_labels'] = []

        def make_labels(BIO, str_words, target: str):

            for i, token in enumerate(BIO):

                if (i == 0) & (token.split('-')[0] == "I"):
                    token = "B-" + token.split('-')[1]

                if token != 'O':
                    bio = token.split('-')
                    position = bio[0]

                    if position == 'B':
                        entity = str_words[i]
                        type = bio[1]
                        span = [i, i]
                        labels = [entity, type, span]
                        item[target].append(labels)

                    elif position == 'I':
                        entity = str_words[i]
                        if (BIO[i - 1].split('-')[0] == 'B') | (BIO[i - 1].split('-')[0] == 'I'):
                            item[target][-1][0] += (' ' + entity)  # entity
                            span = item[target][-1][2]  # span
                            span[1] += 1
                        else:
                            type = bio[1]
                            span = [i, i]
                            labels = [entity, type, span]
                            item[target].append(labels)

        make_labels(gold_BIO, str_words, 'gold_labels')
        make_labels(pred_BIO, str_words, 'pred_labels')

    # make overlap entities
    for i in new_bio_input:
        if (i['pred_labels'] == []) | (i['gold_labels'] == []):
            i['overlapped_entities'] = []

        gold_span_set = [gold_span[-1] for gold_span in i['gold_labels']]
        pred_span_set = [pred_span[-1] for pred_span in i['pred_labels']]

        def has_overlap(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            common_elements = set1.intersection(set2)
            if common_elements:
                return True
            else:
                return False

        overlaps = []
        for pred_span in pred_span_set:
            for gold_span in gold_span_set:

                p_span_list = [i for i in range(pred_span[0], pred_span[1] + 1)]
                g_span_list = [k for k in range(gold_span[0], gold_span[1] + 1)]
                if has_overlap(p_span_list, g_span_list) == True:
                    gold_overlap_ent = i['context'].split()[g_span_list[0]:g_span_list[-1] + 1]
                    pred_overlap_ent = i['context'].split()[p_span_list[0]:p_span_list[-1] + 1]
                    overlaps.append({"gold": (' '.join(gold_overlap_ent), gold_span),
                                     "pred": (' '.join(pred_overlap_ent), pred_span)})
                    i['overlapped_entities'] = overlaps
                elif has_overlap(p_span_list, g_span_list) == False:
                    i['overlapped_entities'] = overlaps

    flatten_bio_input = []
    for item in new_bio_input:
        
        golds = item['gold_labels']
        for k, gold in enumerate(golds):
            true_entity = gold[0]
            true_label = gold[1]
            true_span = gold[2]

            #FN
            if true_span not in [ent['gold'][1] for ent in item['overlapped_entities']]:
                new_dic = {}
                new_dic['type'] = 'FN'
                try:
                    new_dic['id'] = str(item['doc_id']) + '-' + str(k + 1)
                except:
                    new_dic['id'] = str(item['id']) + '-'+ str(k + 1)

                new_dic['context'] = item['context']
                new_dic['true_entity'] = true_entity
                new_dic['true_label'] = true_label
                new_dic['true_span'] = true_span
                new_dic['pred_entity'] = None
                new_dic['pred_label'] = None
                new_dic['pred_span'] = None
                new_dic['gold_BIO'] = item['gold_BIO']
                new_dic['pred_BIO'] = item['pred_BIO']
                flatten_bio_input.append(new_dic) 




            else:
                index = [ent['gold'][1] for ent in item['overlapped_entities']].index(true_span)
                pred_entity = item['overlapped_entities'][index]['pred'][0]
                pred_span = item['overlapped_entities'][index]['pred'][1]
                pred_label = [i[1] for i in item['pred_labels'] if i[0] == pred_entity][0]



                new_dic = {}
                new_dic['type'] = 'TP+FP'

                try:
                    new_dic['id'] = str(item['doc_id']) + '-' + str(k + 1)
                except:
                    new_dic['id'] = str(item['id']) + '-'+ str(k + 1)

                new_dic['context'] = item['context']
                new_dic['true_entity'] = true_entity
                new_dic['true_label'] = true_label
                new_dic['true_span'] = true_span
                new_dic['pred_entity'] = pred_entity
                new_dic['pred_label'] = pred_label
                new_dic['pred_span'] = pred_span
                new_dic['gold_BIO'] = item['gold_BIO']
                new_dic['pred_BIO'] = item['pred_BIO']


                flatten_bio_input.append(new_dic)

        # SPUR
        predictions = item['pred_labels']
        for k, pred in enumerate(predictions):
            pred_entity = pred[0]
            pred_label = pred[1]
            pred_span = pred[2]

            # SPUR
            if pred_span not in [ent['pred'][1] for ent in item['overlapped_entities']]:
                true_entity, true_span, true_label = None, None, None
               
                new_dic = {}
                new_dic['type'] = 'TP+FP'
                try:
                    new_dic['id'] = str(item['doc_id']) + '-' + str(k + 1)
                except:
                    new_dic['id'] = str(item['id']) + '-'+ str(k + 1)

                new_dic['context'] = item['context']
                new_dic['true_entity'] = true_entity
                new_dic['true_label'] = true_label
                new_dic['true_span'] = true_span
                new_dic['pred_entity'] = pred_entity
                new_dic['pred_label'] = pred_label
                new_dic['pred_span'] = pred_span
                new_dic['gold_BIO'] = item['gold_BIO']
                new_dic['pred_BIO'] = item['pred_BIO']
                flatten_bio_input.append(new_dic)
                
                

            else:
                pass

   

    #make ver input

    for j, item in enumerate(flatten_bio_input):
        if item['type'] == 'TP+FP': 

            
            item['pred_span'] = {"start": item['pred_span'][0], "end": item['pred_span'][1]}
            

            if item['true_entity'] == None:  # SPUR
                heu_list = extract_heuristic_cand(item, k=2)
                item['heu_list'] = {}

                for cand in heu_list:
                    cand_entity = list(cand.keys())[0]
                    start = cand[list(cand.keys())[0]][0]
                    end = cand[list(cand.keys())[0]][1]
                    item['heu_list'][cand_entity] = {"start": start, "end": end}

            else:  # TP + FP not spur
                item['true_span'] = {"start": item['true_span'][0], "end": item['true_span'][1]}

                heu_list = extract_heuristic_cand(item, k=2)
                item['heu_list'] = {}

                for cand in heu_list:
                    cand_entity = list(cand.keys())[0]
                    start = cand[list(cand.keys())[0]][0]
                    end = cand[list(cand.keys())[0]][1]
                    item['heu_list'][cand_entity] = {"start": start, "end": end}

        else:  # FN
            pass


    with open("dataset/GENIA_KB.json", "r") as f:
        kb = json.load(f)


    for item in flatten_bio_input:
        if item['type'] == 'TP+FP':
            item['knowledge'] = {}

            cand_list = list(item['heu_list'].keys())
            for cand in cand_list:
                # search KB
                try:
                    item['knowledge'][cand] = kb[cand]
                except:
                    item['knowledge'][cand] = 'no code'


    print("after preprocess:")
    print(flatten_bio_input[0])
    doc_ids = [item['id'].split('-')[0] for item in flatten_bio_input]
    print("num_document:", len(list(set(doc_ids))))

    print()
    print("whole entity:", len(flatten_bio_input))
    print("TP+FP:", len([item for item in flatten_bio_input if item['type'] == 'TP+FP']))
    print("FN:", len([item for item in flatten_bio_input if item['type'] == 'FN']))

    # coverage check

    not_in_KB_idx = []
    in_KB_idx = []
    spr_error_idx = []
    for i, item in enumerate(flatten_bio_input):
        if item['type'] == 'TP+FP':
            if item['true_entity'] != None:

                try:
                    if item['knowledge'][item['true_entity']] != 'no code':
                        in_KB_idx.append(i)
                    else:
                        not_in_KB_idx.append(i)
                except:
                    not_in_KB_idx.append(i)
            else:
                spr_error_idx.append(i)

    return flatten_bio_input


if __name__ == "__main__":
    new_input = preprocess_bio_to_ver_input(bio_input, args.model_name)


    # save
    ver_input = [item for item in new_input if item['type'] == 'TP+FP']
    FN = [item for item in new_input if item['type'] == 'FN']

    with open(args.save_path, "w") as f:
        json.dump(ver_input, f, indent=4)
    with open(args.FN_save_path, "w") as f:
        json.dump(FN, f, indent=4)