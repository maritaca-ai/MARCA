# MARCA - Maritaca AI Research Checklist evAluation
![MARCA](./images/marca_illustration.png)

MARCA is a benchmark for evaluating the performance of AI models in research tasks that require a list of multiple entities. 

All questions and checklists were manually created to ensure coherence. By using an evaluations grounded in the MARCA checklists, we can evaluate the performance of the model more reliably.

## How to run

MARCA uses SERPER to perform web searches and web crawling of pages to the model discretion. You will need to set the SERPER_API_KEY environment variable. You can run evaluations using MARCA with any models with an openai compatible API.

To run a model with MARCA, you can use the following command:

```bash
export SERPER_API_KEY=xxxxx

model_name = "sabia-4"
api_base = "https://chat.maritaca.ai/api"
api_key = "<your key here>"
output_dir = "output/responses_sabia-4"
dataset = "marca_pt|marca_en"
execution_type = "orchestrator|basic"

python main.py --dataset $dataset --model $model --api_base $api_base --api_key $api_key --output_dir $output_dir --k 3 --execution_type $execution_type

```

Note that you can customize various aspect of the run

- dataset : dataset to run the model on. You can use "marca_pt" for the Portuguese dataset or "marca_en" for the English dataset.
- k : number of times to run each question
- execution_type : type of execution to use. You can use "basic" to run the model with the default execution type, "openai_native_web_search" to run the model with the openai native web search tool, or "orchestrator" to run the model with the orchestrator execution type, see the paper for more details.
- orchestrator_model : model to use for the orchestrator in case you want to use a different model for orchestration and other for the subagents. If not provided, the one model will be used.
- subagent_model : model to use for the subagent in case you want to use a different model for orchestration and other for the subagents. If not provided, the model will be used.

## Evaluation

To evaluate the answers of the model, you can use the following command:

```bash
export OPENAI_API_KEY=xx

output_dir = "output/responses_sabia-4"
output_file = "evals/sabia-4.json"

python eval.py --response_dir $output_dir --output_file $output_file
```

The openai key is necessary because we use gpt-4.1-mini to judge each item of the checklists

In the path specified in the --response_dir, you will find the model responses in json format.



