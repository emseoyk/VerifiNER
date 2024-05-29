#!/bin/bash
#sh scripts/verification.sh



# GENIA ConNER #

#entity-level preprocess
python src/preprocess/GENIA_input_preprocess.py \
--data_args_path dataset/BIO_input/GENIA_ConNER_test_BIO.json \
--save_path dataset/verification_input/GENIA_ConNER_type_fact_verification_input.json \
--FN_save_path dataset/verification_input/GENIA_ConNER_FN.json \
--model_name conner

## verification process ## 
#generate explanation  (STEP 2)
python src/chatgpt_async_generate.py \
--prompt prompts/query_candidates/GENIA_type_fact_verification.txt \
--data_args_path dataset/verification_input/GENIA_ConNER_type_fact_verification_input.json \
--type infer_candidate \
--augmentation label_aug \
--dataset_name GENIA \
--save_path dataset/verification_input/GENIA_ConNER_context_rel_verification_input.json

#select final candidate (STEP 3)
python src/chatgpt_async_generate.py \
--prompt prompts/query_candidates/GENIA_contextual_relevance_verification.txt \
--data_args_path dataset/verification_input/GENIA_ConNER_context_rel_verification_input.json \
--type STEP3_self_consistency \
--num_vote 10 \
--dataset_name GENIA \
--save_path dataset/verification_output/GENIA_ConNER_context_rel_verification_output.json

# EValuation (Entity level)
python src/entity_level_eval.py \
--data_args_path dataset/verification_output/GENIA_ConNER_context_rel_verification_output.json \
--FN_data_args_path dataset/verification_input/GENIA_ConNER_FN.json \
--metric_save_path dataset/evaluation/GENIA_ConNER_main_evaluation.txt \
--error_type after_ver_FP_span_idx \
--error_save_path dataset/evaluation/SPAN_error.txt \
--dataset_name GENIA



