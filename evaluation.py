from models.load_model import load_chat_model
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm import tqdm
import json
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BATCH_SIZE = 2

def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1
    
def evaluate_predictions(queries, ground_truths, predictions, evaluation_model):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = INSTRUCTIONS + IN_CONTEXT_EXAMPLES
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n")]
    )
    output_parser = StrOutputParser()
    chain = prompt_template | evaluation_model | output_parser

    messages = []

    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = queries[_idx]
        ground_truth = str(ground_truths[_idx]).strip()
        prediction = prediction.strip()
        
        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()
        
        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue

        messages.append({"query": query, "ground_truth": ground_truth, "prediction": prediction})

    for i in tqdm(range(0, len(messages), BATCH_SIZE)):
        batch = messages[i:i + BATCH_SIZE]
        responses = chain.batch(batch)
        for response in responses:
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results

def evaluate_predictions_2(queries, ground_truths, predictions, domains, static_or_dynamics, question_types, evaluation_model):
    # 初始化统计变量
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    domain_stats = {}
    static_or_dynamic_stats = {}
    question_type_stats = {}
    
    # 系统消息和提示模板设置
    system_message = INSTRUCTIONS + IN_CONTEXT_EXAMPLES
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n")]
    )
    output_parser = StrOutputParser()
    chain = prompt_template | evaluation_model | output_parser

    messages = []

    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = queries[_idx]
        ground_truth = str(ground_truths[_idx]).strip()
        prediction = prediction.strip()
        domain = domains[_idx]
        static_or_dynamic = static_or_dynamics[_idx]
        question_type = question_types[_idx]
        
        # 初始化属性统计字典
        if domain not in domain_stats:
            domain_stats[domain] = {"n_miss": 0, "n_correct": 0, "n_total": 0}
        if static_or_dynamic not in static_or_dynamic_stats:
            static_or_dynamic_stats[static_or_dynamic] = {"n_miss": 0, "n_correct": 0, "n_total": 0}
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {"n_miss": 0, "n_correct": 0, "n_total": 0}
        
        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()
        
        if "i don't know" in prediction_lowercase:
            n_miss += 1
            domain_stats[domain]["n_miss"] += 1
            static_or_dynamic_stats[static_or_dynamic]["n_miss"] += 1
            question_type_stats[question_type]["n_miss"] += 1
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            domain_stats[domain]["n_correct"] += 1
            static_or_dynamic_stats[static_or_dynamic]["n_correct"] += 1
            question_type_stats[question_type]["n_correct"] += 1
        else:
            messages.append({
                "query": query, 
                "ground_truth": ground_truth, 
                "prediction": prediction,
                "domain": domain,
                "static_or_dynamic": static_or_dynamic,
                "question_type": question_type
            })
        
        # 增加统计总数
        domain_stats[domain]["n_total"] += 1
        static_or_dynamic_stats[static_or_dynamic]["n_total"] += 1
        question_type_stats[question_type]["n_total"] += 1

    for i in tqdm(range(0, len(messages), BATCH_SIZE)):
        batch = messages[i:i + BATCH_SIZE]
        responses = chain.batch(batch)
        for idx, response in enumerate(responses):
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1
                msg = batch[idx]
                domain_stats[msg["domain"]]["n_correct"] += 1
                static_or_dynamic_stats[msg["static_or_dynamic"]]["n_correct"] += 1
                question_type_stats[msg["question_type"]]["n_correct"] += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
        "domain_stats": {},
        "static_or_dynamic_stats": {},
        "question_type_stats": {}
    }
    
    # 计算每个属性的统计结果
    for domain, stats in domain_stats.items():
        total = stats["n_total"]
        results["domain_stats"][domain] = {
            "accuracy": stats["n_correct"] / total,
            "missing": stats["n_miss"] / total,
            "hallucination": (total - stats["n_correct"] - stats["n_miss"]) / total
        }

    for static_or_dynamic, stats in static_or_dynamic_stats.items():
        total = stats["n_total"]
        results["static_or_dynamic_stats"][static_or_dynamic] = {
            "accuracy": stats["n_correct"] / total,
            "missing": stats["n_miss"] / total,
            "hallucination": (total - stats["n_correct"] - stats["n_miss"]) / total
        }

    for question_type, stats in question_type_stats.items():
        total = stats["n_total"]
        results["question_type_stats"][question_type] = {
            "accuracy": stats["n_correct"] / total,
            "missing": stats["n_miss"] / total,
            "hallucination": (total - stats["n_correct"] - stats["n_miss"]) / total
        }

    logger.info(results)
    return results


if __name__ == "__main__":
    # Load the model

    api_key = ""
    base_url = ""
    evaluation_model = load_chat_model(model_name="gpt-3.5-turbo", api_key=api_key, base_url=base_url, temperature=0)

    # Evaluate the predictions
    model_name = "gpt-4-turbo"
    knowledge_source = "llm_web"
    predictions_path = f"result/Meta-Llama-3.1-8B-Instruct_llm_kg_predictions_baseline.jsonl"
    with open(predictions_path, "r") as file:
        predictions = [json.loads(line) for line in file]

    queries = [item["question"] for item in predictions]
    ground_truths = [item["ground_truth"] for item in predictions]
    domains = [item["domain"] for item in predictions]
    static_or_dynamics = [item["static_or_dynamic"] for item in predictions]
    question_types = [item["question_type"] for item in predictions]
    predictions = [item["answer"] for item in predictions]
    # evaluate_predictions(queries, ground_truths, predictions, evaluation_model)
    evaluate_predictions_2(queries, ground_truths, predictions, domains, static_or_dynamics, question_types, evaluation_model)
    
    