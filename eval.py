import json
from pydantic import BaseModel
from typing import List
import openai
from textwrap import dedent
import os
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from collections import defaultdict
from dynamic_checklists.dinamic_checklist import available_dynamic_checklists

class CheckItem(BaseModel):
    item_number: int
    explanation: str
    correct: bool


class ChecklistScoreList(BaseModel):
    check_items: List[CheckItem]
    

def check_if_all_checklists_are_present(question_data, checklists):
    for question in question_data:
        checklists_for_question = question["checklist"]
        for checklist_name in checklists_for_question:
            assert checklist_name in checklists, f"Checklist {checklist_name} not found in {checklists.keys()}"
    return True
    
def load_all_categories(dataset):
    categories_data={}
    for i, category in enumerate(os.listdir(os.path.join("questions", dataset))):

        if os.path.isdir(os.path.join("questions", dataset, category)):
            
            questions=json.load(open(os.path.join("questions", dataset, category, "questions.json")))
            checklists=json.load(open(os.path.join("questions", dataset, category, "checklists.json")))
            check_if_all_checklists_are_present(questions, checklists)
            categories_data[category] = {
                "questions": questions,
                "checklists": checklists
            }
    return categories_data


def load_all_responses(response_dir, k=1):
    """Load responses, handling multiple generations per question"""
    qid_to_responses = {}  # Maps original_id to list of responses
    all_json_files = Path(response_dir).glob("**/*.json")
    
    for json_file in all_json_files:
        responses = json.load(open(json_file))
        for response in responses:
            response_id = response["id"]
            
            # Handle both single generation (old format) and multiple generations (new format)
            if k == 1:
                # For backward compatibility with single generation
                original_id = response.get("original_id", response_id)
            else:
                # For multiple generations, use original_id if available, else parse from id
                if "original_id" in response:
                    original_id = response["original_id"]
                else:
                    # Parse original_id from response_id (format: original_id_generation)
                    if "_" in response_id and response_id.split("_")[-1].isdigit():
                        original_id = "_".join(response_id.split("_")[:-1])
                    else:
                        original_id = response_id
            
            if original_id not in qid_to_responses:
                qid_to_responses[original_id] = []
            qid_to_responses[original_id].append(response)
    
    return qid_to_responses


def calculate_generation_stats(all_scores, k):
    """Calculate statistics across multiple generations per question"""
    if k == 1:
        return None
    
    # Group scores by original question ID
    question_to_scores = {}
    for score in all_scores:
        original_id = score["original_id"]
        if original_id not in question_to_scores:
            question_to_scores[original_id] = []
        question_to_scores[original_id].append(score["percentage_correct"])
    
    # Calculate statistics for each question
    question_stats = {}
    for original_id, scores in question_to_scores.items():
        if len(scores) > 0:
            question_stats[original_id] = {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "std": (sum([(x - sum(scores)/len(scores))**2 for x in scores]) / len(scores))**0.5 if len(scores) > 1 else 0,
                "num_generations": len(scores),
                "scores": scores
            }
    
    # Calculate overall generation statistics
    all_means = [stats["mean"] for stats in question_stats.values()]
    all_maxes = [stats["max"] for stats in question_stats.values()]
    all_mins = [stats["min"] for stats in question_stats.values()]
    all_stds = [stats["std"] for stats in question_stats.values()]
    
    overall_stats = {
        "avg_mean_across_questions": sum(all_means) / len(all_means) if all_means else 0,
        "avg_max_across_questions": sum(all_maxes) / len(all_maxes) if all_maxes else 0,
        "avg_min_across_questions": sum(all_mins) / len(all_mins) if all_mins else 0,
        "avg_std_across_questions": sum(all_stds) / len(all_stds) if all_stds else 0,
        "total_questions_with_multiple_generations": len(question_stats)
    }
    
    return {
        "per_question_stats": question_stats,
        "overall_stats": overall_stats
    }

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["marca_pt", "marca_en"], default="marca_pt", help="Dataset to evaluate")
parser.add_argument("--response_dir", type=str, default="responses_sabia31")
parser.add_argument("--output_file", type=str, default="eval_results.json")
parser.add_argument("--k", type=int, default=3, help="Number of generations per question to evaluate")
parser.add_argument("--max_workers", type=int, default=10, help="Number of workers to use for evaluation")
args = parser.parse_args()

response_dir = args.response_dir
categories_data = load_all_categories(args.dataset)
qid_to_responses = load_all_responses(response_dir, args.k)


client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

all_scores = []

avg_score_per_category = {}

def evaluate_single_response(dataset, args):
    """Evaluate a single response and return the score data"""
    question, response_data, checklist_texts, category, client = args
    response_text = response_data["response"]
    response_id = response_data["id"]
    
    checklist_final_text = ""
    for i, checklist_text in enumerate(checklist_texts):
        checklist_final_text += f"{i+1} -> {checklist_text}\n"
        
    system_prompt = dedent(
        """\
    You are a strict and harsh expert evaluator assessing the quality of an answer to a question.
    You will be given a question, the provided response, and a checklist of requirements to be checked in the response.
    
    For each requirement in the checklist, give a brief explanation of why it is satisfied or not, and then state requirement as present or not."""
    )

    user_prompt = dedent(
        """\
    
    Question: {question}
    [END OF QUESTION]
    
    Response: {response}
    [END OF RESPONSE]
    
    Checklist:
    {checklist_final_text}
    [END OF CHECKLIST]"""
    )
    
    formatted_user_prompt = user_prompt.format(
        question=question["question"], response=response_text, checklist_final_text=checklist_final_text
    )
    
    try:
        model_response = client.beta.chat.completions.parse(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_prompt},
            ],
            response_format=ChecklistScoreList,
        )

        checklist_score_list = model_response.choices[0].message.parsed

        total_itens_in_checklist = len(checklist_texts)
        total_items_correct = sum(
            [item.correct for item in checklist_score_list.check_items]
        )

        percentage_correct = total_items_correct / total_itens_in_checklist
        is_75_percent_correct = percentage_correct >= 0.75
        is_100_percent_correct = percentage_correct == 1.0
        checklist_json = checklist_score_list.model_dump()

        return {
            "id": response_id,
            "original_id": question["id"],
            "generation": response_data.get("generation", 1),
            "question": question["question"],
            "response": response_text,
            "checklist": checklist_texts,
            "judged_checklist": checklist_json,
            "percentage_correct": percentage_correct,
            "category": category,
            "is_75_percent_correct": is_75_percent_correct,
            "is_100_percent_correct": is_100_percent_correct,
        }
        
    except Exception as e:
        print(f"Error evaluating response {response_id}: {e}")
        return None


# Prepare all evaluation tasks
evaluation_tasks = []

total_token_usage = defaultdict(int)

for category, data in categories_data.items():
    questions = data["questions"]
    category_checklists = data["checklists"]
    
    for question in questions:
        qid = question["id"]
        
        # Check if we have responses for this question
        if qid not in qid_to_responses:
            print(f"Warning: No responses found for question {qid}")
            continue
            
        responses_for_question = qid_to_responses[qid]
        checklists_for_question = question["checklist"]
        
        for response_data in responses_for_question:
            for key, value in response_data["token_usage_details"].items():
                total_token_usage[key] += value

        checklist_texts = []
        for checklist_name in checklists_for_question:
            checklist_definition = category_checklists[checklist_name.lower()]
            for checklist_element in checklist_definition:
                if isinstance(checklist_element, str):
                    checklist_texts.append(checklist_element)
                elif isinstance(checklist_element, dict):
                    execution_class_name=checklist_element["class"]
                    execution_parameters=checklist_element["parameters"]
                    dynamic_template_class=available_dynamic_checklists[execution_class_name]
                    dynamic_template_texts=dynamic_template_class.get_checklist(**execution_parameters)
                    checklist_texts.extend(dynamic_template_texts)
                else:
                    raise ValueError(f"Invalid checklist element type: {type(checklist_element)}, {checklist_element}")

        # Add tasks for each response
        for response_data in responses_for_question:
            evaluation_tasks.append((question, response_data, checklist_texts, category, client))

# Execute evaluations in parallel
max_workers = args.max_workers  # Adjust based on your needs and API rate limits
category_scores = {}

print(f"Starting parallel evaluation of {len(evaluation_tasks)} responses...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_task = {executor.submit(evaluate_single_response, args.dataset, task): task for task in evaluation_tasks}
    
    # Process completed tasks
    for future in tqdm(as_completed(future_to_task), total=len(evaluation_tasks), desc="Evaluating responses"):
        task = future_to_task[future]
        question, response_data, checklist_texts, category, _ = task
        
        try:
            result = future.result()
            if result is not None:
                all_scores.append(result)
                
                # Track scores by category and question
                qid = question["id"]
                if category not in category_scores:
                    category_scores[category] = {}
                if qid not in category_scores[category]:
                    category_scores[category][qid] = []
                category_scores[category][qid].append({
                    "percentage_correct": result["percentage_correct"],
                    "is_75_percent_correct": result["is_75_percent_correct"],
                    "is_100_percent_correct": result["is_100_percent_correct"],
                })
                
        except Exception as e:
            print(f"Error processing future for question {question['id']}: {e}")

# Calculate category averages
for category, questions_scores in category_scores.items():
    all_values = []
    for qid, scores in questions_scores.items():
        all_values.extend(scores)
    
    if all_values:  # Only add if we have scores
        # Get total questions for this category
        total_questions = len(categories_data[category]["questions"])
        
        avg_score_per_category[category] = {
            "avg_score": sum([score["percentage_correct"] for score in all_values]) / len(all_values),
            "avg_75_percent_correct": sum([score["is_75_percent_correct"] for score in all_values]) / len(all_values),
            "avg_100_percent_correct": sum([score["is_100_percent_correct"] for score in all_values]) / len(all_values),
            "scores": questions_scores,
            "total_responses": len(all_values),
            "total_questions": total_questions
        }

if all_scores:
    average_percentage_correct = sum(
        [score["percentage_correct"] for score in all_scores]
    ) / len(all_scores)
    
    average_75_percent_correct = sum([score["is_75_percent_correct"] for score in all_scores]) / len(all_scores)
    average_100_percent_correct = sum([score["is_100_percent_correct"] for score in all_scores]) / len(all_scores)

    print(f"Total responses evaluated: {len(all_scores)}")
    print(f"Average percentage correct: {average_percentage_correct:.4f}")
    print(f"Average 75% correct: {average_75_percent_correct:.4f}")
    print(f"Average 100% correct: {average_100_percent_correct:.4f}")
    
    if args.k > 1:
        total_questions = len(set(score["original_id"] for score in all_scores))
        print(f"Total unique questions: {total_questions}")

    avg_score_per_category["all_categories"] = {
        "avg_score": average_percentage_correct,
        "avg_75_percent_correct": average_75_percent_correct,
        "avg_100_percent_correct": average_100_percent_correct,
        "total_responses": len(all_scores),
        "total_questions": sum([cat_data["total_questions"] for cat_data in avg_score_per_category.values()])
    }
    

    # Calculate generation statistics if k > 1
    generation_stats = calculate_generation_stats(all_scores, args.k)
    
    all_results = {
        "avg_score_per_category": avg_score_per_category,
        "all_scores": all_scores,
        "k": args.k,
        "generation_stats": generation_stats,
        "total_token_usage": total_token_usage,
    }

    with open(args.output_file, "w") as f:
        f.write(json.dumps(all_results, indent=4, ensure_ascii=False))

    print("Done!")
else:
    print("No scores generated - check if responses were found and evaluation was successful")
