import re, time, math
from tqdm import tqdm
import openai, anthropic
from mistralai.client import MistralClient
from concurrent.futures import ThreadPoolExecutor

def request_by_client(client, prompt, model, max_retry=5, temperature=0., top_p=0.9, max_token=None, stop_token_ids=None, max_completion_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    retry = 0
    # only used in openai style api for local open source model inference served by vllm library
    extra_body = {"stop_token_ids": stop_token_ids} if stop_token_ids else None 
    while True:
        try:
            if isinstance(client, openai.OpenAI):
                if model == 'o1-mini' or model == 'o1-preview':
                    max_tokens, temperature, top_p = None, 1., 1. # o1 model does not support temperature and top_p not equal to 1 
                else:
                    max_completion_tokens=None
                completion = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=max_token,
                                max_completion_tokens=max_completion_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                extra_body=extra_body
                )
                output = completion.choices[0].message.content
                break
            elif isinstance(client, anthropic.Anthropic):
                response = client.messages.create(
                    model=model,
                    max_tokens=max_token,
                    temperature=temperature,
                    messages=messages,
                    top_p=top_p
                )
                output = response.content[0].text
                break
            elif isinstance(client, MistralClient):
                response = client.chat(
                    model=model,
                    messages=messages,
                    max_tokens=max_token,
                    temperature=temperature,
                    top_p=top_p
                )
                output = response.choices[0].message.content
                break
            else:
                raise NotImplementedError(f"Unsupported client type {type(client)} detected, please consider add your own client impl here ....")
        except Exception as e:
            if retry < max_retry:
                print(f"Exception occurred, wait 3s: {e}", flush=True)
                time.sleep(3)
            else:
                output = ''
                break
            retry += 1
    return output


def parse_model_response(model_response):
    sol_corr, fst_err_step, err_reason = '', '', ''
    # use the format specified in the prompt to parse response
    try:
        # incase the model generate more than one response, we select the first one
        sol_analysis, rest_part = model_response.split('Solution Correctness:')[:2]
        sol_corr, rest_part = rest_part.split('First Error Step:')[:2]
        fst_err_step, err_reason = rest_part.split('Error Reason:')[:2]
    except Exception as e:
        print(f'Fail to parse model response: {e}')

    return sol_corr, fst_err_step, err_reason


def single_thread_response_generation(data, client, model_name, temperature, top_p, max_tokens, stop_token_ids, max_completion_tokens, EVAL_KEY):
    prompt = data['Query_Prompt']
    model_response = request_by_client(client=client, 
                                    prompt=prompt, 
                                    model=model_name,
                                    temperature=temperature,
                                    top_p=top_p, 
                                    max_token=max_tokens,
                                    stop_token_ids=stop_token_ids,
                                    max_completion_tokens=max_completion_tokens)
    sol_corr, fst_err_step, err_reason = parse_model_response(model_response)
    data[EVAL_KEY] = {
        'evaluated_model': model_name,
        'evaluation_raw_response': model_response,
        'solution_correctness': sol_corr,
        'first_error_step': fst_err_step,
        'error_reason': err_reason, 
    }


def multi_thread_response_generation(data_list, client, model_name, temperature, top_p, max_tokens, stop_token_ids, max_completion_tokens, EVAL_KEY, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(lambda x: single_thread_response_generation(x, client, model_name, temperature, top_p, max_tokens, stop_token_ids, max_completion_tokens, EVAL_KEY), data_list),
                total=len(data_list)
            )
        )


def single_thread_scoring(data, client, subject, model="gpt-4-turbo"):
    prompt = generate_scoring_prompt(data, subject)
    # by default we use gpt4 turbo for scoring in greedy sampling setting
    response = request_by_client(client, prompt, model=model)
    try:
        reasoning = re.search(r"Step-by-Step Reasoning:\s*(.*?)\s*Student Error Reason Analysis:", response, re.DOTALL).group(1)
        error_analysis = re.search(r"Student Error Reason Analysis:\s*(.*?)\s*Final Decision:", response, re.DOTALL).group(1)
        final_decision = re.search(r"Final Decision:\s*(.*)", response, re.DOTALL).group(1)
        data['Error_Reason_Correctness_Analysis'] = {
            'scoring_model': model,
            'scoring_raw_response': response,
            'annotation_analysis': reasoning,
            'error_reason_analysis': error_analysis,
            'error_reason_correctness': final_decision
        }
    except Exception as e:
        data['Error_Reason_Correctness_Analysis'] = {
            'scoring_model': model,
            'scoring_raw_response': response,
            'annotation_analysis': 'ERROR_PARSING',
            'error_reason_analysis': 'ERROR_PARSING',
            'error_reason_correctness': 'ERROR_PARSING'
        }
    return

def multi_thread_scoring(unscored_data_list, client, subject, model_name, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(lambda x: single_thread_scoring(x, client, subject, model_name), unscored_data_list),
                total=len(unscored_data_list)
            )
        )


def generate_scoring_prompt(data, subject):
    if subject != 'coding':
        prompt = f"""As an experienced {data['Subject']} teacher, your assistance is required to evaluate a student's explanation regarding the error in a problem solution. The task involves a detailed understanding of the problem, the incorrect solution provided, and the ground truth behind the error. Your analysis should focus on whether the student's explanation aligns with the actual error in the solution.

Please find the details below:

- Question: {data['Question']}
- Incorrect Solution Provided: {data['Model_Solution_Steps']}
- First Incorrect Step in the Solution: {data['Model_Solution_First_Error_Step']}
- Ground Truth Error Reasons: {data['Model_Solution_Error_Reason']}
- Ground Truth Rectified Steps: {data['Model_Solution_Rectified_First_Error_Step']}
- Student's Explanation of the Error: {data['Evaluation_Result']['error_reason']}

Based on this information, please provide the following:

1. Step-by-Step Reasoning: [Offer a succinct, step-by-step interpretation of the ground truth error reason.]
2. Student Error Reason Analysis: [Analyze the student's explanation step by step, determining its accuracy in reflecting the actual error briefly.]
3. Final Decision: [State only 'Correct' or 'Wrong' to conclude whether the student's explanation correctly identifies the error based on your analysis.]

Please follow this format without any additional introductory or concluding statements."""
    else:
        prompt = f"""As an experienced programmer, your assistance is required to evaluate a student's explanation regarding the error in a coding problem solution. The task involves a detailed understanding of the problem, the incorrect solution provided, and the ground truth error reason. Your analysis should focus on whether the student's explanation aligns with the actual error in the solution.

Please find the details below:

- Question: {data['Question']}
- Incorrect Solution Provided: {data['Model_Solution_Steps']}
- First Incorrect Step in the Solution: {data['Model_Solution_First_Error_Step']}
- Ground Truth Error Reasons: {data['Model_Solution_Error_Reason']}
- Ground Truth Rectified Steps: {data['Model_Solution_Rectified_First_Error_Step']}
- Student's Reported First Error Step: {data['Evaluation_Result']['first_error_step']}
- Student's Explanation of the Error: {data['Evaluation_Result']['error_reason']}

Based on this information, please provide the following:

1. Step-by-Step Reasoning: [Offer a succinct, step-by-step interpretation of the ground truth error reason and error step.]
2. Student Error Reason Analysis: [Analyze the first error step reported by student and its explanation step by step, determining its accuracy in reflecting/aligning with the ground truth error briefly.]
3. Final Decision: [State only 'Correct' or 'Wrong' to conclude whether the student's explanation correctly identifies the error location and reason based on your analysis.]

Please follow this format without any additional introductory or concluding statements."""
    return prompt


def process_subject_data(subject_data, k_shot, kshot_data):
    # Assign subject specific prompts to every single 
    for uid_key in subject_data:
        for sol in subject_data[uid_key]:
            if sol['Subject'] == 'coding':
                if k_shot == 0:
                    k_shot_demo = ''
                else:
                    k_shot_demo = f"Followings are the {k_shot}-shot examples for your reference:\n\n"
                    for demo in kshot_data['coding'][:k_shot]:
                        demo_string =f"""\
Question: 
{demo['Question']}

Code solution: 
{demo['Solution']}

{demo['cot_analysis']}
"""
                        k_shot_demo += demo_string
                prompt = f"""Following is a coding question and its solution. Your task is to examine the solutions step by step and determine the solution correctness.
If the solution is incorrect, please further find out the first error line and explain the error reason. 

Following are the specific definitions of the fields:
Solution Correctness: Determines if the code correctly solves the problem. The code must be executable, logically sound, and correctly answer the question.  
First Error Step: Examine each line to categorize as correct, neutral, or incorrect. Correct lines execute well and contribute to solving the problem. Neutral lines are auxiliary and not critical to the main logic. Incorrect lines contain logical, implementation errors, or show a misunderstanding of the problem. Both explanatory comments and code lines could be considered as correct, neutral or incorrect. If you believe the error is caused by missing logic, report the first reasonable place you think the missing code should be inserted.
Error Reason: For the first incorrect line/location identified, explain the specific mistakes and provide a corrected version of the line. The revised line could be either a explanatory comment or a code line.

{k_shot_demo}
Below is the question and solution for you to solve:
Question: 
{sol['Question']}

Code solution: 
{sol['Model_Solution_Steps']}


Please follow the desired response format:
Solution Analysis: [Give a step by step analysis on the solution correctness here]
Solution Correctness: [Input 'correct'/'incorrect' here to indicate the overall correctness of the solution]
First Error Step: [Input the first incorrect line of code here. Input 'N/A' if the solution is correct.]
Error Reason: [Input the error reason and the rectified reasoning of the first error line here. Input 'N/A' if the solution is correct.] 

Please follow this format without any additional introductory or concluding statements.
"""
            else:
                if k_shot == 0:
                    k_shot_demo = ''
                else:
                    k_shot_demo = f"\nFollowings are the {k_shot}-shot examples for your reference:\n\n"
                    for demo in kshot_data[sol['Subject']][:k_shot]:
                        k_shot_demo += f"Question: {demo['Question']}\n"
                        k_shot_demo += f"Options: {demo['Options']}\n"
                        k_shot_demo += f"Step by Step Solution: {demo['Model_Solution_Steps']}\n"
                        k_shot_demo += f"{demo['cot_analysis']}\n\n"
                prompt = f"""Following is a question/solution pair in subject {sol['Subject']}. Your task is to examine the solutions step by step and determine the solution correctness.
If the solution is incorrect, please further find out the first error step and explain the error reason. 

Following are the specific definitions of the fields:
Solution Correctness: Does the solution correctly answer the question with justifiable reasoning and selected the corrected options?  
First Error Step: For every step it can either be correct, neutral or incorrect. Correct steps are those that possess sound logic and correct computation and lead to the correct answer.
Neutral steps are those step that are explanatory, exploring or focusing on background illustration. They have no obvious mistakes but is not very clear if they lead to the correct answer.
Incorrect steps are those with factual errors, computation errors or understanding/logic errors. These steps might or might not detour the reasoning path to incorrect answers. 
We need to single out the first step that comes with above errors or lead to incorrect answers. 
Error Reason: For the identified first error step, please specify the errors made in this step and suggest a rectified reasoning step instead.

{k_shot_demo}
Below is the question and solution for you to solve:
Question: {sol['Question']}
Options: {sol['Options']}
Step by Step Solution: {sol['Model_Solution_Steps']}


Please follow the desired response format:
Solution Analysis: [Give a step by step analysis on the solution correctness here]
Solution Correctness: [Input 'correct'/'incorrect' here to indicate the overall correctness of the solution]
First Error Step: [Input 'Step x' here to indicate the first error step here. Input 'N/A' if the solution is correct.]
Error Reason: [Input the error reason and the rectified reasoning of the first error step here. Input 'N/A' if the solution is correct.] 

Please follow this format without any additional introductory or concluding statements.
"""
            sol['Query_Prompt'] = prompt   


def MCC_score(tp, tn, fp, fn):
    numerator = tp*tn-fp*fn
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) 
    if numerator == 0 or denominator == 0: return 0. 
    return numerator/denominator

def mr_score(model_stat, w1=0.2, w2=0.3, w3=0.5):
    mcc_score = MCC_score(model_stat['t1-tp'], model_stat['t1-tn'], model_stat['t1-fp'], model_stat['t1-fn'])
    mr_score_auto = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy-auto']
    return mr_score_auto

def construct_eval_stats(basic_stats):
    # true positive and true negative of task1-numbers that correctly determine the solution correctness
    t1_tp, t1_tn = basic_stats['t1-tp'], basic_stats['t1-tn']
    # correct number of task 2-determine the first error step
    t2_corr_num = basic_stats['t2_corr_num']
    # correct number of task 3-determine the error reason, judged either by annotators or GPT4
    t3_corr_num_auto = basic_stats['t3_corr_num_auto']
    # correct and incorrect solution numbers in the given subject 
    correct_sol_num, incorrect_sol_num = basic_stats['correct_sol_num'], basic_stats['incorrect_sol_num']
    final_stats = {
            't1-tp': t1_tp,
            't1-tn': t1_tn,
            't1-fp': (incorrect_sol_num-t1_tn),
            't1-fn': (correct_sol_num-t1_tp),
            't1-recall': t1_tp/correct_sol_num,
            't1-precision': t1_tp/(t1_tp+incorrect_sol_num-t1_tn), # precision = tp/(tp+fp)
            't2-accuracy': t2_corr_num/incorrect_sol_num,
            't3-accuracy-auto': t3_corr_num_auto/incorrect_sol_num,
    }
    return final_stats

def get_mr_score(basic_subject_stat):
    eval_subject_stat = construct_eval_stats(basic_subject_stat)
    # We have grid searched on the weights over all the evaluation results we have. The following ratio is selected due to the consideration on the 
    # differentiation of MR-Scores and the substantiality of each task. 
    return mr_score(eval_subject_stat, w1=0.2, w2=0.3, w3=0.5)
