import argparse, os, json
from openai import OpenAI
from script.utils import (multi_thread_scoring, 
                            multi_thread_response_generation,
                            process_subject_data,
                            get_mr_score)

SUBJECTS = [
    'biology',
    'math',
    'physics',
    'medicine',
    'coding',
    'chemistry',
    'logic'
]
EVAL_KEY = 'Evaluation_Result'

def load_dataset(dataset_path, k_shot, demo_path):    
    kshot_data = None
    if k_shot != 0:
        with open(demo_path) as file:
            kshot_data = json.load(file)
    # load corresponding subjects and construct corresponding evaluation prompt
    loaded_dataset = {}
    if os.path.isdir(dataset_path):
        for subject in SUBJECTS:
            if os.path.exists(f"{dataset_path}/{subject}.json"):
                print(f'Loading subject {subject} data ......')
                with open(f"{dataset_path}/{subject}.json") as file:
                    subject_data = json.load(file)
                process_subject_data(subject_data, k_shot, kshot_data)   
                loaded_dataset[subject] = subject_data
    return loaded_dataset

def generate_response(client, benchmark, model_name, temperature, top_p, max_tokens, stop_token_ids, unscored_save_path, max_workers):
    results = {}
    for subject in benchmark:
        results[subject] = []
        print(f"Generating answers of {subject} ...... ")
        all_queries = []
        if type(benchmark[subject]) == list:
            for qs_dict in benchmark[subject]:
                if EVAL_KEY in qs_dict and not qs_dict[EVAL_KEY]["evaluation_raw_response"]:
                    all_queries.append(qs_dict)
        else:
            for question_uuid in benchmark[subject]:
                for qs_dict in benchmark[subject][question_uuid]:
                    all_queries.append(qs_dict)
        # these query dict should be modified and contain the desired response results
        multi_thread_response_generation(all_queries, 
                                         client, 
                                         model_name, 
                                         temperature, 
                                         top_p, 
                                         max_tokens, 
                                         stop_token_ids, 
                                         EVAL_KEY,
                                         max_workers)
        if type(benchmark[subject]) == list:
            results[subject] = benchmark[subject]
        else:
            results[subject] = all_queries 
        with open(unscored_save_path, 'w') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
    return results

def score_error_reason(score_client, score_model, eval_results, scored_save_path, max_workers): 
    step_mapper = {f"step {i}": f"{i}"  for i in range(30)}
    for subject in eval_results:
        for data in eval_results[subject]:
            # We only need to score incorrect solutions with correctly predicted first error step
            data['Need_Error_Reason_Review'] = False
            if data['Model_Solution_Correctness'] == 'incorrect':
                if data[EVAL_KEY]['solution_correctness'].strip().lower() == 'incorrect':
                    if subject == 'coding':
                        # for coding task, the first error step is only a rough indicator and should be scored by scoring model or annotator
                        data['Need_Error_Reason_Review'] = True
                        continue
                    if data[EVAL_KEY]['first_error_step'].strip().isdigit():
                        error_step_pred = data[EVAL_KEY]['first_error_step'].strip()
                    elif data[EVAL_KEY]['first_error_step'].strip().lower() in step_mapper:
                        error_step_pred = step_mapper[data[EVAL_KEY]['first_error_step'].strip().lower()]
                    else:
                        error_step_pred = ''
                    if error_step_pred == data['Model_Solution_First_Error_Step']:
                        data['Need_Error_Reason_Review'] = True
    # score with gpt4 
    for subject in eval_results:
        print(f"scoring answers of {subject} ......")
        to_be_scored_data = []
        for data in eval_results[subject]:
            # save the gpt4 score results so that we can recover from any breaks without re-querying
            if (data['Need_Error_Reason_Review'] and 'Error_Reason_Correctness_Analysis' not in data) or \
                ('Error_Reason_Correctness_Analysis' in data and not data['Error_Reason_Correctness_Analysis']['scoring_raw_response']):
                to_be_scored_data.append(data)
        # by default we use gpt4 turbo for scoring in greedy sampling setting
        multi_thread_scoring(to_be_scored_data, score_client, subject, score_model, max_workers)
        with open(scored_save_path, 'w') as file:
            json.dump(eval_results, file, indent=2, ensure_ascii=False)
    return eval_results

def calculate_mr_score(scored_eval_results):
    mr_score_stats, mr_scores = {}, {}
    step_mapper = {f"step {i}": f"{i}"  for i in range(30)}
    for subject in scored_eval_results:
        task1_true_positive, task1_true_negative = 0, 0
        correct_sol_num, incorrect_sol_num = 0, 0 
        task2_accy, task3_accy_auto = 0, 0    
        for data in scored_eval_results[subject]:
            if data['Model_Solution_Correctness'] == 'correct':
                correct_sol_num +=1
            else:
                incorrect_sol_num +=1 
            correctness_pred = data[EVAL_KEY]['solution_correctness'].strip().lower()
            if data['Model_Solution_Correctness'] == correctness_pred:
                if data['Model_Solution_Correctness'] == 'correct':
                    task1_true_positive +=1
                else:
                    task1_true_negative +=1
                    # only if the solution is incorrect and the model agrees on the incorrectness do 
                    # we look into task2 and task3 performance. Note for coding task, it is hard to pinpoint
                    # the exact location of the first error step. We instead use the line as a rough indicator 
                    # and leave the judgement of the error reason to the scoring model or annotator. 
                    if subject == 'coding':
                        if 'correct' in data['Error_Reason_Correctness_Analysis']['error_reason_correctness'].lower():
                            task2_accy += 1
                            task3_accy_auto +=1
                            continue
                    if data[EVAL_KEY]['first_error_step'].strip().isdigit():
                        error_step_pred = data[EVAL_KEY]['first_error_step']
                    elif data[EVAL_KEY]['first_error_step'].strip().lower() in step_mapper:
                        error_step_pred = step_mapper[data[EVAL_KEY]['first_error_step'].strip().lower()]
                    else:
                        error_step_pred = ''
                    if error_step_pred == data['Model_Solution_First_Error_Step']:
                        task2_accy += 1
                        if 'correct' in data['Error_Reason_Correctness_Analysis']['error_reason_correctness'].lower():
                            task3_accy_auto +=1
        mr_score_stats[subject] = {
            't1-tp': task1_true_positive,
            't1-tn': task1_true_negative,
            't2_corr_num': task2_accy,
            't3_corr_num_auto': task3_accy_auto,
            'correct_sol_num': correct_sol_num,
            'incorrect_sol_num': incorrect_sol_num
        }
    for subject in mr_score_stats:
        mr_scores[subject] = get_mr_score(mr_score_stats[subject])
    return mr_scores
            


def main(args):
    if '/' in args.eval_model_name:
        # name of open-sourced model served by vllm is the absolute path of the downloaded model folder
        succint_model_name = args.eval_model_name.split('/')[-1]
    else:
        succint_model_name = args.eval_model_name # commercial models
    unscored_save_path = f"{args.output_dir}/{succint_model_name}_{args.shot_num}shot_cot_True_eval_results.json"
    scored_save_path = f"{args.output_dir}/{succint_model_name}_{args.shot_num}shot_cot_True_scored_eval_results.json"    
    eval_client = OpenAI(base_url=args.eval_base_url, api_key=args.eval_api_key)
    # load the MR-BEAN dataset and construct the corresponding evaluation prompts
    mr_bean_dataset = load_dataset(args.dataset_path, args.shot_num, args.demo_path)
    try:
        with open(unscored_save_path) as file:
            generated_res = json.load(file)
        for subject in generated_res:
            mr_bean_dataset[subject] = generated_res[subject]
        print("Cached eval results found, reusing generated subject data ....")  
    except Exception as e:
        print("No cached eval result found, proceed to full dataset evaluation ....")
    mr_bean_eval_results = generate_response(eval_client, 
                                             mr_bean_dataset,
                                             args.eval_model_name, 
                                             args.temperature,
                                             args.top_p,
                                             args.max_tokens,
                                             args.stop_token_ids,
                                             unscored_save_path,
                                             args.max_workers)
    score_client = OpenAI(base_url=args.score_base_url, api_key=args.score_api_key)
    scored_mr_bean_eval_results = score_error_reason(score_client, args.score_model_name, mr_bean_eval_results, scored_save_path, args.max_workers)
    mr_scores = calculate_mr_score(scored_mr_bean_eval_results)
    sum_mr_score = 0
    for key in mr_scores:
        print(f"{key}: {round(mr_scores[key]*100, 1)}") 
        sum_mr_score += mr_scores[key]
    print('Average MR-Score:', round(sum_mr_score*100/len(mr_scores), 1))
    return mr_scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate open sourced models on MR.BEAN benchmark")
    parser.add_argument('--eval_base_url', type=str, required=True, help='The base url to the openAI-api compatible server of the evaluated model')
    parser.add_argument('--eval_api_key', type=str, required=False, help='The potential api-key to the api server of the evaluated model', default='placeholder')
    parser.add_argument('--eval_model_name', type=str, required=True, help='The name of the evaluated model, for local open-sourced model please provide absolute path to the model') 
    parser.add_argument('--score_base_url', type=str, required=True, help='The base url to the openAI-api compatible server for scoring the error reason')
    parser.add_argument('--score_api_key', type=str, required=False, help='The potential api-key to the api server for scoring the error reason', default='')
    parser.add_argument('--score_model_name', type=str, required=True, help='The name of the scoring model. We recommend using gpt-4-turbo')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the MR.BEAN dataset directory')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory for saving evaluation results')
    parser.add_argument('--temperature', '-t', type=float, required=False, default=0.0,  help='Temperature for sampling')
    parser.add_argument('--top_p', '-p', type=float, required=False, default=0.9, help='Top-p threshold for sampling')
    parser.add_argument('--max_tokens', '-m', type=int, required=False, default=None, help='Max token numbers to generate during sampling')
    parser.add_argument('--stop_token_ids', type=int, required=False,  nargs="+", help='List of stop token ids because default tokenizer used by vllm might not using correct stop tokens in chat models.')
    parser.add_argument('--shot_num', '-k', type=int, required=False, default=0, help='The number of demonstrations for evaluated model')
    parser.add_argument('--demo_path', type=str, required=False, default='', help='The path to the few shot demo file for evaluation')
    parser.add_argument('--max_workers', type=int, required=False, default=5, help='The number of multi-thread workers for api requests')
    args = parser.parse_args()
    main(args)
