# RAG-X

This repository is based on the 🥈 [solution](https://github.com/USTCAGI/CRAG-in-KDD-Cup2024) of [Meta KDD Cup '24 CRAG: Comphrensive RAG Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) !

For more information, visit our official homepage at https://ustc-rag-x.github.io/, and you can read our paper at https://arxiv.org/abs/2409.13694.

### 📊 Dataset
The CRAG dataset, central to the KDD Cup 2024, is particularly well-suited for real-world retrieval-based Question Answering (QA) tasks due to its diverse query types and document formats. To enhance its utility, we have significantly processed the original dataset, specifically converting the HTML-formatted web pages containing external knowledge into markdown (MD) format using Jina. This transformation makes the data more accessible and easier for large language models (LLMs) to utilize effectively. 
Our refined version of the dataset, named RM3QA—'A Real-time Multi-domain, Multi-format, Multi-source Question Answer Dataset for RAG'—is now available [here](https://huggingface.co/datasets/fishsure/RM3QA).

## 🏁 Our RAG framework

![framework1_00](./README.assets/framework1_00.png)

Our RAG framework, RAG-X, employs an LLM Agent-based Router to intelligently select relevant knowledge sources from a curated mix of structured and unstructured data, including web pages and a mock API. Our retrieval process avoids web searches, focusing exclusively on the provided information sources. The retrieval process is divided into three steps: broad retrieval narrows down vast external knowledge, focused retrieval uses sparse, dense, or hybrid methods to identify key information, and rank refinement ensures the final output is accurate and prioritized.  After retrieval, we enhance the LLM's reasoning with noise chunks, Chain of Thought (CoT), and In-Context Learning (ICL), leading to more  relevant responses.

### Router

We designed two specialized routers: the **Domain Router** and the **Dynamism Router**.

We trained **Sentence Classifiers** as routers for Domain and Dynamism based on **Llama3-8B-Instruct**.

Model Weights locate at `models/router`.

### Retriever

#### Web Pages

1. **Broad Retrieval** : Segment text from Web Pages into chunks of **1024** tokens. Use **BM25** to select the top **50** relevant text blocks.
2. **Focused Retrieval**: Segment text blocks into **256**-token chunks, transform them into embeddings using the `bge-m3` model, and select the top 10 relevant chunks based on cosine similarity.
3. **Rank Refinement**: Use `bge-m3-v2-reranker` to re-rank the Top **10** chunks and select the Top **3** segments.

For more detail, refers to `models/retrieve/retriever.py`.

#### Mock APIs

1. **Named Entity Recognition (NER)**: Use Llama3-70B to classify named entities in questions into predefined categories specific to each domain.
2. **Entity Match**: Match extracted entities with API input parameters (e.g., converting company names to ticker symbols for finance APIs).
3. **Time Information Extraction**: Extract temporal information from user inputs and compute relative time based on query time.
4. **API Select**: Use manually designed rules to select relevant APIs for a given question.
5. **Json to Markdown**: Convert JSON output from APIs into Markdown format for better processing by LLMs.

For more detail, refers to `models/mock_api`.

### Generation



#### Adaptive Few-Shot CoT Prompt

```python
"""For the given question and multiple references from Mock API, think step by step, then provide the final answer.
Current date: {query_time}

Note: 
- For your final answer, please use as few words as possible. 
- The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    - `what's the latest score update for OKC's game today?` (There is no game for OKC today)
    - `how many times has curry won the nba dunk contest?` (Steph Curry has never participated in the NBA dunk contest)
- If you don't know the answer, you MUST respond with `I don't know`
- If the references do not contain the necessary information to answer the question, respond with `I don't know`
- Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`
- Your output format needs to meet the requirements: First, start with `## Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `## Final Answer\n` and using as few words as possible.

### Question
{query}

### References
{references}
"""
```



## Get Started

### Install Dependencies!
```bash
pip install -r requirements.txt
```

### Download Models

`meta-llama/Meta-Llama-3-8B-Instruct`:

+ https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
+ https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct

`BAAI/bge-m3`:

+ https://huggingface.co/BAAI/bge-m3
+ https://www.modelscope.cn/models/Xorbits/bge-m3

`BAAI/bge-reranker-v2-m3`:

+ https://huggingface.co/BAAI/bge-reranker-v2-m3
+ https://www.modelscope.cn/models/AI-ModelScope/bge-reranker-v2-m3

### Run LLM

+ Use API server(such as openai)

  + open [`main.py`](main.py)
  + setup your `api_key`and `base_url`
  + choose your `model_name`(default: gpt-4o)

+ Run LLM locally with vLLM server

  + download your llm(such as Llama3-70B)

  + start your server using python

    ```bash
    python -m vllm.entrypoints.openai.api_server \
      --model ./meta-llama/Meta-Llama-3-8B-Instruct \
      --served-model-name Llama3-8B \
      --trust-remote-code
    ```

    refer to https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html for more details

  + open [`main.py`](main.py)

  + choose you `model_name`, such as `Llama3-8B`

  + add parameter `stop` to function `load_model`

    ```python
    load_model(model_name=model_name, api_key=api_key, base_url=base_url, temperature=0, stop=["<|eot_id|>"])
    ```

+ Run LLM locally with Ollama

  + install ollama

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

  + pull llm(such as llama3)

    ```
    ollama pull llama3
    ```

  + open [`main.py`](main.py)

  + choose you `model_name`, such as `llama3`

  + use `load_model_ollama`



### Predict

run `main.py`

### Evaluate

run `evaluation.py`

