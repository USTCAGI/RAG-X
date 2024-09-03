import os
import bz2
import json
from tqdm import tqdm
from loguru import logger
import argparse

from models.load_model import load_chat_model
from models.retrieve.retriever import Retriever, Retriever_Milvus
from models.model import RAGModel, RAGModel_2Stage, RAGModel_3Stage

def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": [], "domain": [], "static_or_dynamic": [], "question_type": []}

    try:
        if dataset_path.endswith(".bz2"):
            with bz2.open(dataset_path, "rt", encoding='utf-8') as file:
                batch = initialize_batch()
                for line in file:
                    try:
                        item = json.loads(line)
                        for key in batch:
                            batch[key].append(item[key])
                            # print(key)
                        
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                    except json.JSONDecodeError:
                        logger.warn("Warning: Failed to decode a line.")
                # Yield any remaining data as the last batch
                if batch["query"]:
                    yield batch
        else:
            with open(dataset_path, "r") as file:
                batch = initialize_batch()
                for line in file:
                    try:
                        item = json.loads(line)
                        for key in batch:
                            batch[key].append(item[key])
                        
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                    except json.JSONDecodeError:
                        logger.warn("Warning: Failed to decode a line.")
                # Yield any remaining data as the last batch
                if batch["query"]:
                    yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e
    
def generate_predictions(dataset_path, participant_model, batch_size):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions, contexts = [], [], [], []
    domains, static_or_dynamics, question_types= [], [], []

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions, batch_contexts = participant_model.batch_generate_answer(batch)
        queries.extend(batch["query"])
        domains.extend(batch["domain"])
        static_or_dynamics.extend(batch["static_or_dynamic"])
        question_types.extend(batch["question_type"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
        contexts.extend(batch_contexts)
        logger.info(f"Query Example: {queries[-1]}")
        logger.info(f"Ground Truth Example: {ground_truths[-1]}")
        logger.info(f"Prediction Example: {predictions[-1]}")
    return queries, ground_truths, predictions, contexts , domains, static_or_dynamics, question_types

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--api_key", type=str, default="<your-api-key>")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B-Instruct") 
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--embedding_model_path", type=str, default="models/retrieve/embedding_models/bge-m3")
    parser.add_argument("--reranker_model_path", type=str, default="models/retrieve/reranker_models/bge-reranker-v2-m3") 
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparse", type=int, default=0)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--knowledge_source", type=str, default="llm_web")
    parser.add_argument("--prompt_type", type=str, default="base")
    parser.add_argument("--dataset_path", type=str, default="task1_split_0_no_link.jsonl.bz2") 
    parser.add_argument("--icl", type=int, default=0)
    parser.add_argument("--broad_retrieval", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    # Set the environment variable for the mock API
    os.environ["CRAG_MOCK_API_URL"] = "http://localhost:8001"

    args = parse_arg()
    # Load the model
    api_key = args.api_key
    # base_url = "http://210.45.70.162:28083/v1/"
    base_url = args.base_url
    model_name = args.model_name
    temperature = args.temperature
    top_p = args.top_p
    chat_model = load_chat_model(model_name=model_name, api_key=api_key, base_url=base_url, temperature=temperature, top_p=top_p, max_tokens=100)

    # Load the retriever
    embedding_model_path = args.embedding_model_path
    reranker_model_path = args.reranker_model_path
    top_k = args.top_k
    top_n = args.top_n
    rerank = args.rerank
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    sparse = args.sparse
    device = args.device
    broad_retrieval = args.broad_retrieval

    retriever = Retriever(top_k, top_n, embedding_model_path, reranker_model_path, rerank, chunk_size, chunk_overlap, sparse, broad_retrieval,  device)
    # To use the retriever with Milvus, uncomment the following lines and comment the previous line
    # collection_name = "bge_m3_crag_task_1_dev_v3_llamaindex"
    # uri = "http://localhost:19530"
    # # uri = ".models/retrieve/milvus.db"
    # # retriever = Retriever_Milvus(10, 5, collection_name, uri, embedding_model_path, reranker_model_path, rerank=True)
    # retriever = Retriever_Milvus(5, 5, collection_name, uri, embedding_model_path, reranker_model_path, rerank=False)
    knowledge_source = args.knowledge_source
    prompt_type = args.prompt_type
    icl = args.icl
    rag_model = RAGModel(chat_model, retriever, knowledge_source, prompt_type)
    # rag_model = RAGModel_2Stage(chat_model, retriever, knowledge_source)
    # rag_model = RAGModel_3Stage(chat_model, retriever, knowledge_source)
    # Generate predictions
    dataset_path = args.dataset_path
    queries, ground_truths, predictions, contexts, domains, static_or_dynamics, question_types  = generate_predictions(dataset_path, rag_model, args.batch_size)
    
    # Save the predictions
    output_path = f"result/{model_name}_{knowledge_source}_predictions.jsonl"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w") as file:
        for query, ground_truth, prediction, context, domain, static_or_dynamic, question_type in zip(queries, ground_truths, predictions, contexts , domains, static_or_dynamics, question_types ):
            item = {"question": query, "answer": prediction, "ground_truth": ground_truth, "contexts": context, "domain": domain, "static_or_dynamic": static_or_dynamic, "question_type": question_type}
            file.write(json.dumps(item) + "\n")
    
    logger.info(f"Predictions saved to {output_path}.") 