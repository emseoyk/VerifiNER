import json
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--FN_data_args_path", type=str, required=True)
parser.add_argument("--metric_save_path", type=str, required=True)
parser.add_argument("--error_type", type=str, required=True)
parser.add_argument("--error_save_path", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()



def common_value(before, after):
    common_values = [value for value in before if value in after]
    return len(common_values), common_values


def error_check(error_case_idx, output_file_path):
    error_name = error_case_idx
    output_string = ""

    idx_category = eval(error_case_idx)
    for i in range(len(idx_category)):
        output_string += f"\n<<{i}th error case of {str(error_name)}>>\n\n"
        output_string += f"     * true entity&label: {(ver_result[idx_category[i]]['true_entity'], ver_result[idx_category[i]]['true_label'])}\n\n"
        output_string += f"     * before_verification_prediction: {(ver_result[idx_category[i]]['pred_entity'], ver_result[idx_category[i]]['pred_label'])}\n\n"
        output_string += f"     * verification result: {ver_result[idx_category[i]]['final_answer']}\n"

        if idx_category == parsing_error_idx:
            output_string += "\n     ㄴparsing_error_idx\n"
            output_string += f"     ㄴindex: {ver_result[idx_category[i]]['id']}\n"
            output_string += f"     ㄴoutput: {ver_result[idx_category[i]]['revision']}\n"
        else:
            output_string += f"\n     ㄴindex: {ver_result[idx_category[i]]['id']}\n"
            output_string += f"\n     ㄴcontext: {ver_result[idx_category[i]]['context']}\n"
            output_string += f"     ㄴcand_list: {ver_result[idx_category[i]]['cand_list']}\n"

            explanations = ver_result[idx_category[i]]['per_cand_explanations']
            for j, exp in enumerate(explanations):
                output_string += f"            ㄴ**{j + 1}th candidate**\n"
                output_string += f"             ㄴcandidate: {exp['cand']}\n"

                knowledge_without_newlines = exp['verbalized_knowledge'].replace('\n', ' ')
                output_string += f"             ㄴknowledge: {knowledge_without_newlines}\n"

                output_string += f"             ㄴcand_exp_pair: {exp['cand_exp_pair']}\n\n"
            

    # Save the output to a text file
    with open(output_file_path, 'w') as file:
        file.write(output_string)
    
    return output_string
    

# load output result
with open(args.data_args_path, "r") as f:
    ver_result = json.load(f)



if __name__ == "__main__":

    # False Negative
    with open(args.FN_data_args_path, "r") as f:
        FN_items = json.load(f)

    if args.dataset_name == "GENIA":
        # sampling FN_items based on ver_result 
        ver_result_doc_ids = list(set([item['id'].split('-')[0] for item in ver_result]))
        sampled_FN_items = []
        for item in FN_items:
            if item['id'].split('-')[0] in ver_result_doc_ids:
                sampled_FN_items.append(item)

        FN_items = sampled_FN_items 
    else:
        pass
    
    ##################################################################################

    before_ver_TP = 0
    before_ver_FP_spur = 0
    before_ver_FP_label = 0
    before_ver_FP_span = 0
    before_ver_FP_both = 0

    before_ver_TP_idx = []
    before_ver_FP_spur_idx = []
    before_ver_FP_label_idx = []
    before_ver_FP_span_idx = []
    before_ver_FP_both_idx = []

    for i, item in enumerate(ver_result):
        # before verification
        gold_entity = item['true_entity']
        gold_label = item['true_label']
        pred_entity = item['pred_entity']
        pred_label = item['pred_label']

        # before ver - TP
        if (gold_entity == pred_entity) & (gold_label == pred_label):
            before_ver_TP += 1
            before_ver_TP_idx.append(i)

        elif gold_entity is None:
            before_ver_FP_spur += 1
            before_ver_FP_spur_idx.append(i)

        elif (gold_entity == pred_entity) & (gold_label != pred_label):
            before_ver_FP_label += 1
            before_ver_FP_label_idx.append(i)

        elif (gold_entity != pred_entity) & (gold_label == pred_label):
            before_ver_FP_span += 1
            before_ver_FP_span_idx.append(i)

        elif (gold_entity != pred_entity) & (gold_label != pred_label):
            before_ver_FP_both += 1
            before_ver_FP_both_idx.append(i)
    ##################################################################################
    # after verification
    after_ver_TN = 0
    after_ver_TP = 0
    after_ver_FN = 0
    after_ver_FP_spur = 0
    after_ver_FP_label = 0
    after_ver_FP_span = 0
    after_ver_FP_both = 0

    after_ver_TN_idx = []
    after_ver_TP_idx = []
    after_ver_FN_idx = []
    after_ver_FP_spur_idx = []
    after_ver_FP_label_idx = []
    after_ver_FP_span_idx = []
    after_ver_FP_both_idx = []

    parsing_error = 0
    TP2parse = 0
    FP2parse = 0
    parsing_error_idx = []

    for i, item in enumerate(ver_result):
        gold_entity = item['true_entity']
        gold_label = item['true_label']

        if item['final_answer'] == 'parsing_error':
            if (gold_entity == item['pred_entity']) & (gold_label == item['pred_label']):
                TP2parse += 1
            else:
                FP2parse += 1

            parsing_error += 1
            parsing_error_idx.append(i)
            continue

        # after ver
        if (gold_entity is None) & (item['final_answer'][-1] == 'None'):
            after_ver_TN += 1
            after_ver_TN_idx.append(i)
            continue

        elif (gold_entity is not None) & (item['final_answer'][-1] == 'None'):
            after_ver_FN += 1
            after_ver_FN_idx.append(i)
            continue

        elif (gold_entity is None) & (item['final_answer'][-1] != 'None'):
            after_ver_FP_spur += 1
            after_ver_FP_spur_idx.append(i)
            continue

        verified_entity = item['final_answer'][0]
        verified_label = item['final_answer'][1]

        if (gold_entity == verified_entity) & (gold_label == verified_label):
            after_ver_TP += 1
            after_ver_TP_idx.append(i)

        elif (gold_entity == verified_entity) & (gold_label != verified_label):
            after_ver_FP_label += 1
            after_ver_FP_label_idx.append(i)

        elif (gold_entity != verified_entity) & (gold_label == verified_label):
            after_ver_FP_span += 1
            after_ver_FP_span_idx.append(i)

        elif (gold_entity != verified_entity) & (gold_label != verified_label):
            after_ver_FP_both += 1
            after_ver_FP_both_idx.append(i)




    after_ver_FP = after_ver_FP_spur + after_ver_FP_label + after_ver_FP_span + after_ver_FP_both
    ##################################################################################

    # idx list
    tp2tp, tp2tp_idx = common_value(before_ver_TP_idx, after_ver_TP_idx)
    tp2fn, tp2fn_idx = common_value(before_ver_TP_idx, after_ver_FN_idx)
    tp2fp_spur, tp2fp_spur_idx = common_value(before_ver_TP_idx, after_ver_FP_spur_idx)
    tp2fp_label, tp2fp_label_idx = common_value(before_ver_TP_idx, after_ver_FP_label_idx)
    tp2fp_span, tp2fp_span_idx = common_value(before_ver_TP_idx, after_ver_FP_span_idx)
    tp2fp_both, tp2fp_both_idx = common_value(before_ver_TP_idx, after_ver_FP_both_idx)

    fp_spur2tn, fp_spur2tn_idx = common_value(before_ver_FP_spur_idx, after_ver_TN_idx)
    fp_label2tn, fp_label2tn_idx = common_value(before_ver_FP_label_idx, after_ver_TN_idx)
    fp_span2tn, fp_span2tn_idx = common_value(before_ver_FP_span_idx, after_ver_TN_idx)
    fp_both2tn, fp_both2tn_idx = common_value(before_ver_FP_both_idx, after_ver_TN_idx)

    fp_spur2fn, fp_spur2fn_idx = common_value(before_ver_FP_spur_idx, after_ver_FN_idx)
    fp_label2fn, fp_label2fn_idx = common_value(before_ver_FP_label_idx, after_ver_FN_idx)
    fp_span2fn, fp_span2fn_idx = common_value(before_ver_FP_span_idx, after_ver_FN_idx)
    fp_both2fn, fp_both2fn_idx = common_value(before_ver_FP_both_idx, after_ver_FN_idx)

    fp_spur2tp, fp_spur2tp_idx = common_value(before_ver_FP_spur_idx, after_ver_TP_idx)
    fp_label2tp, fp_label2tp_idx = common_value(before_ver_FP_label_idx, after_ver_TP_idx)
    fp_span2tp, fp_span2tp_idx = common_value(before_ver_FP_span_idx, after_ver_TP_idx)
    fp_both2tp, fp_both2tp_idx = common_value(before_ver_FP_both_idx, after_ver_TP_idx)

    fp_spur2fp_spur, fp_spur2fp_spur_idx = common_value(before_ver_FP_spur_idx, after_ver_FP_spur_idx)
    fp_label2fp_spur, fp_label2fp_spur_idx = common_value(before_ver_FP_label_idx, after_ver_FP_spur_idx)
    fp_span2fp_spur, fp_span2fp_spur_idx = common_value(before_ver_FP_span_idx, after_ver_FP_spur_idx)
    fp_both2fp_spur, fp_both2fp_spur_idx = common_value(before_ver_FP_both_idx, after_ver_FP_spur_idx)

    fp_spur2fp_label, fp_spur2fp_label_idx = common_value(before_ver_FP_spur_idx, after_ver_FP_label_idx)
    fp_label2fp_label, fp_label2fp_label_idx = common_value(before_ver_FP_label_idx, after_ver_FP_label_idx)
    fp_span2fp_label, fp_span2fp_label_idx = common_value(before_ver_FP_span_idx, after_ver_FP_label_idx)
    fp_both2fp_label, fp_both2fp_label_idx = common_value(before_ver_FP_both_idx, after_ver_FP_label_idx)

    fp_spur2fp_span, fp_spur2fp_span_idx = common_value(before_ver_FP_spur_idx, after_ver_FP_span_idx)
    fp_label2fp_span, fp_label2fp_span_idx = common_value(before_ver_FP_label_idx, after_ver_FP_span_idx)
    fp_span2fp_span, fp_span2fp_span_idx = common_value(before_ver_FP_span_idx, after_ver_FP_span_idx)
    fp_both2fp_span, fp_both2fp_span_idx = common_value(before_ver_FP_both_idx, after_ver_FP_span_idx)

    fp_spur2fp_both, fp_spur2fp_both_idx = common_value(before_ver_FP_spur_idx, after_ver_FP_both_idx)
    fp_label2fp_both, fp_label2fp_both_idx = common_value(before_ver_FP_label_idx, after_ver_FP_both_idx)
    fp_span2fp_both, fp_span2fp_both_idx = common_value(before_ver_FP_span_idx, after_ver_FP_both_idx)
    fp_both2fp_both, fp_both2fp_both_idx = common_value(before_ver_FP_both_idx, after_ver_FP_both_idx)

    ##################################################################################
    before_ver_FP = before_ver_FP_spur  + before_ver_FP_label  + before_ver_FP_span + before_ver_FP_both
    before_ver_FN = len(FN_items)
   
    before_ver_precision = before_ver_TP / (before_ver_FP + before_ver_TP) 
    before_ver_recall= before_ver_TP / (before_ver_TP + before_ver_FN)
    # ##############
    # before_ver_recall = 1
    # ##############



    after_ver_TP = (tp2tp  + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp)
    after_ver_TN = fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn
    after_ver_FP = (tp2fp_spur + tp2fp_label + tp2fp_span +tp2fp_both +fp_spur2fp_spur+fp_label2fp_spur+fp_span2fp_spur
    +fp_both2fp_spur
    +fp_spur2fp_label
    +fp_label2fp_label
    +fp_span2fp_label
    +fp_both2fp_label
    +fp_spur2fp_span
    +fp_label2fp_span
    +fp_span2fp_span+fp_both2fp_span
    +fp_spur2fp_both
    +fp_label2fp_both
    +fp_span2fp_both
    +fp_both2fp_both)


    after_ver_FN = tp2fn +fp_spur2fn+fp_label2fn+fp_span2fn+fp_both2fn

    ##################################################################################
    
    precision = after_ver_TP / (after_ver_TP + after_ver_FP)
    recall = after_ver_TP / (after_ver_TP + after_ver_FN + before_ver_FN )
    f1 = 2 / (1/precision + 1/recall)

    ##################################################################################
    # KB gold coverage check
    true_not_in_cand_list_idx = []
    true_not_in_cand_list = []
    else_list = []

    for idx in after_ver_FP_span_idx:
        cand_list = ver_result[idx]['cand_list']
        true_entity = ver_result[idx]['true_entity']
        if true_entity not in cand_list:
            true_not_in_cand_list_idx.append(idx)
            true_not_in_cand_list.append((true_entity, cand_list))
        else:
            else_list.append((true_entity, cand_list))

    # both error
    for idx in after_ver_FP_both_idx:
        cand_list = ver_result[idx]['cand_list']
        true_entity = ver_result[idx]['true_entity']
        if true_entity not in cand_list:
            true_not_in_cand_list_idx.append(idx)
            true_not_in_cand_list.append((true_entity, cand_list))


        else:
            else_list.append((true_entity, cand_list))

    minus = 0
    i = 0
    k = 0

    ##################################################################################
    # whole coverage check

    not_in_KB_idx = []
    in_KB_idx = []
    spr_error_idx = []
    for i, item in enumerate(ver_result):
        if item['true_entity'] != None:

            try:
                cand_list = item['cand_list']
                if item['true_entity'] not in cand_list:
                    not_in_KB_idx.append(i)
                else:
                    in_KB_idx.append(i)
            except:
                not_in_KB_idx.append(i)
        else:
            spr_error_idx.append(i)
    #################################################################################

    metric_output_string = ''
    metric_output_string += f"whole_prediction: {len(ver_result)} = {before_ver_TP + before_ver_FP_spur + before_ver_FP_label + before_ver_FP_span + before_ver_FP_both}\n\n"
    metric_output_string += f"before_ver_TP: {before_ver_TP}\n"
    metric_output_string += f"before_ver_FP: {before_ver_FP_spur + before_ver_FP_label + before_ver_FP_span + before_ver_FP_both}\n"
    metric_output_string += f"     ㄴbefore_ver_FP_spur: {before_ver_FP_spur}\n"
    metric_output_string += f"     ㄴbefore_ver_FP_label: {before_ver_FP_label}\n"
    metric_output_string += f"     ㄴbefore_ver_FP_span: {before_ver_FP_span}\n"
    metric_output_string += f"     ㄴbefore_ver_FP_both: {before_ver_FP_both}\n\n"

    metric_output_string += f"whole_verification: {len(ver_result)} = {parsing_error + after_ver_TP + after_ver_TN + after_ver_FN + after_ver_FP_spur + after_ver_FP_label + after_ver_FP_span + after_ver_FP_both}\n\n"
    metric_output_string += f" ㄴafter_ver_TP: {after_ver_TP}\n"
    metric_output_string += f" ㄴafter_ver_TN: {after_ver_TN}\n"
    metric_output_string += f" ㄴafter_ver_FN: {after_ver_FN}\n"
    after_ver_FP = after_ver_FP_spur + after_ver_FP_label + after_ver_FP_span + after_ver_FP_both
    metric_output_string += f" ㄴafter_ver_FP: {after_ver_FP}\n\n"

    metric_output_string += f"     ㄴafter_ver_FP_spur: {after_ver_FP_spur}\n"
    metric_output_string += f"     ㄴafter_ver_FP_label: {after_ver_FP_label}\n"
    metric_output_string += f"     ㄴafter_ver_FP_span: {after_ver_FP_span}\n"
    metric_output_string += f"     ㄴafter_ver_FP_both: {after_ver_FP_both}\n"
    metric_output_string += f"parsing_error: {parsing_error}\n"
    metric_output_string += f"     ㄴTP2parse_error: {TP2parse}\n"
    metric_output_string += f"     ㄴFP2parse_error: {FP2parse}\n\n"

    metric_output_string += f"< prediction True (TP->) >\n"
    metric_output_string += f"before verification TP: {before_ver_TP} = {TP2parse + tp2tp + tp2fn + tp2fp_spur + tp2fp_label + tp2fp_span + tp2fp_both}\n\n"
    metric_output_string += f"TP->TP: {tp2tp} , {tp2tp / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"TP->FN: {tp2fn} , {tp2fn / (before_ver_FP + before_ver_TP) * 100}\n"

    metric_output_string += f"TP->FP_spur: {tp2fp_spur} , {tp2fp_spur / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"TP->FP_label: {tp2fp_label} , {tp2fp_label / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"TP->FP_span: {tp2fp_span} , {tp2fp_span / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"TP->FP_both: {tp2fp_both} , {tp2fp_both / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n"
    metric_output_string += f"TP->FP: {tp2fp_spur + tp2fp_label + tp2fp_span + tp2fp_both} , {(tp2fp_spur + tp2fp_label + tp2fp_span + tp2fp_both) / (before_ver_FP + before_ver_TP) * 100}\n\n"
    metric_output_string += "\n"
    metric_output_string += f"TP -> parsing error: {TP2parse} , {TP2parse / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n"
    metric_output_string += f"TP -> False: {tp2fp_spur + tp2fp_label + tp2fp_span + tp2fp_both + tp2fn} , {(tp2fp_spur + tp2fp_label + tp2fp_span + tp2fp_both + tp2fn) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"before verification FP: {before_ver_FP} = {(FP2parse) + (fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn) + (fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp) + (fp_spur2fn + fp_label2fn + fp_span2fn + fp_both2fn) + fp_spur2fp_spur + fp_label2fp_spur + fp_span2fp_spur + fp_both2fp_spur + fp_spur2fp_label + fp_label2fp_label + fp_span2fp_label + fp_both2fp_label + fp_spur2fp_span + fp_label2fp_span + fp_span2fp_span + fp_both2fp_span + fp_spur2fp_both + fp_label2fp_both + fp_span2fp_both + fp_both2fp_both}\n"
    metric_output_string += f"FP_spur -> TN: {fp_spur2tn} , {fp_spur2tn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> TN: {fp_label2tn} , {fp_label2tn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> TN: {fp_span2tn} , {fp_span2tn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> TN: {fp_both2tn} , {fp_both2tn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> TP: {fp_spur2tp} , {fp_spur2tp / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> TP: {fp_label2tp} , {fp_label2tp / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> TP: {fp_span2tp} , {fp_span2tp / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> TP: {fp_both2tp} , {fp_both2tp / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"
    metric_output_string += f"FP->True: {fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp} , {(fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> FN: {fp_spur2fn} , {fp_spur2fn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> FN: {fp_label2fn} , {fp_label2fn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> FN: {fp_span2fn} , {fp_span2fn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> FN: {fp_both2fn} , {fp_both2fn / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP -> FN: {fp_spur2fn + fp_label2fn + fp_span2fn + fp_both2fn} , {(fp_spur2fn + fp_label2fn + fp_span2fn + fp_both2fn) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> FP_spur: {fp_spur2fp_spur} , {fp_spur2fp_spur / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> FP_spur: {fp_label2fp_spur} , {fp_label2fp_spur / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> FP_spur: {fp_span2fp_spur} , {fp_span2fp_spur / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> FP_spur: {fp_both2fp_spur} , {fp_both2fp_spur / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP -> SPUR: {fp_spur2fp_spur + fp_label2fp_spur + fp_span2fp_spur + fp_both2fp_spur} , {(fp_spur2fp_spur + fp_label2fp_spur + fp_span2fp_spur + fp_both2fp_spur) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> FP_label: {fp_spur2fp_label} , {fp_spur2fp_label / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> FP_label: {fp_label2fp_label} , {fp_label2fp_label / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> FP_label: {fp_span2fp_label} , {fp_span2fp_label / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> FP_label: {fp_both2fp_label} , {fp_both2fp_label / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP -> LABEL: {fp_spur2fp_label + fp_label2fp_label + fp_span2fp_label + fp_both2fp_label} , {(fp_spur2fp_label + fp_label2fp_label + fp_span2fp_label + fp_both2fp_label) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> FP_span: {fp_spur2fp_span} , {fp_spur2fp_span / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> FP_span: {fp_label2fp_span} , {fp_label2fp_span / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> FP_span: {fp_span2fp_span} , {fp_span2fp_span / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> FP_span: {fp_both2fp_span} , {fp_both2fp_span / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP -> SPAN: {fp_spur2fp_span + fp_label2fp_span + fp_span2fp_span + fp_both2fp_span} , {(fp_spur2fp_span + fp_label2fp_span + fp_span2fp_span + fp_both2fp_span) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP_spur -> FP_both: {fp_spur2fp_both} , {fp_spur2fp_both / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_label -> FP_both: {fp_label2fp_both} , {fp_label2fp_both / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_span -> FP_both: {fp_span2fp_both} , {fp_span2fp_both / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP_both -> FP_both: {fp_both2fp_both} , {fp_both2fp_both / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"FP -> BOTH: {fp_spur2fp_both + fp_label2fp_both + fp_span2fp_both + fp_both2fp_both} , {(fp_spur2fp_both + fp_label2fp_both + fp_span2fp_both + fp_both2fp_both) / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += "\n\n"

    metric_output_string += f"FP -> parsing error: {FP2parse} , {FP2parse / (before_ver_FP + before_ver_TP) * 100}\n\n"

    metric_output_string += f"FP -> False: {fp_spur2fn + fp_label2fn + fp_span2fn + fp_both2fn + fp_spur2fp_spur + fp_label2fp_spur + fp_span2fp_spur + fp_both2fp_spur + fp_spur2fp_label + fp_label2fp_label + fp_span2fp_label + fp_both2fp_label + fp_spur2fp_span + fp_label2fp_span + fp_span2fp_span + fp_both2fp_span + fp_spur2fp_both + fp_label2fp_both + fp_span2fp_both + fp_both2fp_both + FP2parse} , {(fp_spur2fn + fp_label2fn + fp_span2fn + fp_both2fn + fp_spur2fp_spur + fp_label2fp_spur + fp_span2fp_spur + fp_both2fp_spur + fp_spur2fp_label + fp_label2fp_label + fp_span2fp_label + fp_both2fp_label + fp_spur2fp_span + fp_label2fp_span + fp_span2fp_span + fp_both2fp_span + fp_spur2fp_both + fp_label2fp_both + fp_span2fp_both + fp_both2fp_both + FP2parse) / (before_ver_FP + before_ver_TP) * 100}\n\n"

    metric_output_string += "\n"

    metric_output_string += f"parsing_error: {parsing_error} , {parsing_error / (before_ver_FP + before_ver_TP) * 100}\n"

    metric_output_string += "\n\n"

    metric_output_string += f"Whole pred entity : {before_ver_FP + before_ver_TP}\n"

    metric_output_string += "\n\n"

    metric_output_string += f"before_ver_TP ratio: {before_ver_TP} , {before_ver_TP / (before_ver_FP + before_ver_TP) * 100}\n"
    metric_output_string += f"before_ver_FP ratio: {before_ver_FP} , {before_ver_FP / (before_ver_FP + before_ver_TP) * 100}\n"

    metric_output_string += f"before_ver_FN: {before_ver_FN}\n\n"

    metric_output_string += f"before_ver_precision : {before_ver_precision}\n"
    metric_output_string += f"before_ver_recall : {before_ver_recall}\n"

    try: 
        metric_output_string += f"before_ver_f1: {2 / ((1 / before_ver_precision) + (1 / before_ver_recall))}\n"
    except:
        metric_output_string += "CrossDataset setting == before_ver_precision = 0"

    metric_output_string += "\n\n"

    metric_output_string += f"after_ver_T ratio: {(tp2tp + fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp)} , {(tp2tp + fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp) / (before_ver_FP + before_ver_TP)}\n"
    metric_output_string += f"after_ver_F ratio: {(before_ver_FP + before_ver_TP - (tp2tp + fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp))} , {(before_ver_FP + before_ver_TP - (tp2tp + fp_spur2tn + fp_label2tn + fp_span2tn + fp_both2tn + fp_spur2tp + fp_label2tp + fp_span2tp + fp_both2tp)) / (before_ver_FP + before_ver_TP)}\n"

    metric_output_string += f"after_ver_precision: {precision}\n"
    metric_output_string += f"after_ver_recall: {recall}\n"
    metric_output_string += f"after_ver_f1: {f1}\n"

    metric_output_string += "\n\n\n"


    with open(args.metric_save_path, "w", encoding="utf-8") as file:
        file.write(metric_output_string)
    #################################################################################

    if args.error_type != None:
        error_check(args.error_type, args.error_save_path)
        
    ##################################################################################