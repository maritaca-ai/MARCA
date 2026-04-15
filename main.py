from models.models import AssistantWithTools, OpenaiNativeWebSearch, OrchestratorModel, SubagentModel
from tool.tools import SearchTool, PageScraperTool, SubagentTool
import argparse
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def get_all_categories(dataset):
    return os.listdir(os.path.join("questions", dataset))


def load_all_questions(dataset,categories):
    all_questions = []
    for category in categories:
        question_dir = os.path.join("questions", dataset, category)
        questions_file = os.path.join(question_dir, "questions.json")
        with open(questions_file, "r") as f:
            questions = json.load(f)
        for question in questions:
            question["category"] = category
            all_questions.append(question)
    return all_questions


def process_single_question_generation(question, k, model, max_retries=3):
    """Process a single question generation attempt"""
    question_text = question["question"]
    id = question["id"]
    category = question["category"]
    
    for i in range(max_retries):
        try:
            messages = [
                {"role": "user", "content": question_text}
            ]

            final_response, conversation_generated, total_token_usage = model.generate_response(
                messages
            )
            error = None
            break
        except Exception as e:
            print(f"try {i} Error for question {id}, generation {k+1}: {e}")
            if i == max_retries - 1:
                print(f"Failed to get response for question {id}, generation {k+1} after {max_retries} tries")
                final_response = "Error"
                error = str(e)
                conversation_generated = []
                total_token_usage = {}
            continue

    # Create unique ID for each generation
    response_id = f"{id}_{k+1}" if args.k > 1 else id
    
    # For orchestrator models, conversation_generated is actually full_history
    # Store it appropriately for debugging
    result = {
        "id": response_id,
        "original_id": id,
        "generation": k+1,
        "question": question_text,
        "response": final_response,
        "conversation_generated": conversation_generated,
        "error": error,
        "category": category,
        "token_usage_details": total_token_usage,
    }
    
    # If this is an orchestrator model, also store full_history separately for clarity
    if isinstance(model, OrchestratorModel):
        result["full_history"] = conversation_generated
    
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["marca_pt", "marca_en"], default="marca_pt")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
parser.add_argument("--api_key", type=str, default="KEY")
parser.add_argument("--category", type=str, default="all")
parser.add_argument("--execution_type", type=str, default="basic", choices=["basic", "openai_native_web_search", "orchestrator"])
parser.add_argument("--orchestrator_model", type=str, default=None, help="Model name for orchestrator (if different from --model)")
parser.add_argument("--subagent_model", type=str, default=None, help="Model name for subagent (if different from --model)")
parser.add_argument("--output_dir", type=str)
parser.add_argument("--k", type=int, default=3, help="Number of times to generate each question")
parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers")
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature for the model",
)
parser.add_argument(
    "--reasoning_effort",
    type=str,
    default=None,
    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    help="Optional reasoning effort level for the model",
)
args = parser.parse_args()


system_prompt = """You are a helpful research assistant. You have access to the internet to research and fullfil the user's request."""

categories = []
if args.category == "all":
    categories = get_all_categories(args.dataset)
else:
    categories = [args.category]

available_tools = [SearchTool(), PageScraperTool()]


if args.execution_type == "basic":
    model = AssistantWithTools(
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        tools=available_tools,
        system_prompt=system_prompt,
        reasoning_effort=args.reasoning_effort,
    )
elif args.execution_type == "openai_native_web_search":
    model = OpenaiNativeWebSearch(
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
elif args.execution_type == "orchestrator":
    # Create subagent model
    subagent_model_name = args.subagent_model if args.subagent_model else args.model
    subagent_system_prompt = """You are a helpful research assistant. You have access to the internet to research and fullfil the user's request
Use these tools to find accurate, up-to-date information to answer the query you receive. Be thorough and provide detailed answers."""
    
    subagent_model = SubagentModel(
        model_name=subagent_model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        tools=available_tools,
        system_prompt=subagent_system_prompt,
    )
    
    # Create subagent tool
    subagent_tool = SubagentTool(subagent_model)
    
    # Create orchestrator model
    orchestrator_model_name = args.orchestrator_model if args.orchestrator_model else args.model
    orchestrator_system_prompt = """You are an orchestrator responsible for breaking down complex queries into smaller, more manageable sub-queries.
When you receive a query, analyze it and break it down into focused sub-queries that can be answered independently. 
Be aware that you can create new sub-queries based on the information you receive from the subagents, your objective is to be as complete as possible.
For example, for a original question such as 'who were the top 5 most popular actors of the year 2025, how old are each of them?' you can first ask the subagent to find the top 5 most popular actors of the year 2025, and then ask the subagent to find the age of each of them separately.
Use the delegate_to_subagent function to send each sub-query to a subagent that will search for information and provide answers.
After receiving answers from subagents, synthesize the information to provide a comprehensive final answer to the user's original query."""
    
    model = OrchestratorModel(
        model_name=orchestrator_model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        system_prompt=orchestrator_system_prompt,
        subagent_tool=subagent_tool,
    )

# Load all questions from all categories
all_questions = load_all_questions(args.dataset,categories)

# Create list of all tasks (question, generation_number)
all_tasks = []
for question in all_questions:
    for k in range(args.k):
        all_tasks.append((question, k))

# Process all questions in parallel
all_responses = []

with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    # Submit all tasks
    future_to_task = {
        executor.submit(process_single_question_generation, question, k, model): (question, k)
        for question, k in all_tasks
    }
    
    # Process completed tasks with progress bar
    for future in tqdm(as_completed(future_to_task), total=len(all_tasks)):
        question, k = future_to_task[future]
        try:
            response = future.result()
            all_responses.append(response)
        except Exception as e:
            print(f"Error processing question {question['id']}, generation {k+1}: {e}")
            # Create error response
            response_id = f"{question['id']}_{k+1}" if args.k > 1 else question['id']
            all_responses.append({
                "id": response_id,
                "original_id": question['id'],
                "generation": k+1,
                "question": question['question'],
                "response": "Error",
                "conversation_generated": [],
                "error": str(e),
                "category": question['category'],
            })


# Group responses by category and save
responses_by_category = {}
for response in all_responses:
    category = response["category"]
    if category not in responses_by_category:
        responses_by_category[category] = []
    # Remove category from response before saving
    response_to_save = {k: v for k, v in response.items() if k != "category"}
    responses_by_category[category].append(response_to_save)

# Save responses for each category
for category, responses in responses_by_category.items():
    category_output_dir = os.path.join(args.output_dir, category)
    output_file = os.path.join(category_output_dir, "responses.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)
