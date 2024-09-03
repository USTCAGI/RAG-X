from typing import Any, Dict, List
from prompts.templates import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from models.mock_api.api import MockAPI
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import json
import os
import sys
from collections import defaultdict
from json import JSONDecoder
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger
from sentence_transformers import SentenceTransformer
from utils.cragapi_wrapper import CRAG

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, chat_model, retriever, knowledge_source, prompt_type="base", icl=None):
        self.chat_model = chat_model
        self.retriever = retriever
        self.knowledge_source = knowledge_source
        SYSTEM_PROMPT = "You are a helpful assistant."
        BASELINE_SYSTEM_PROMPT = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        self.prompt_type = prompt_type
        self.icl = icl
        if icl == 0:
            self.invalid_questions_examples  = {
                #都置为空，作为对照
                "open": "",
                "finance": "",
                "music": "",
                "movie": "",
                "sports": ""
            }
        elif icl == 1:
            self.invalid_questions_examples  = {
                "open": "-`when did hamburg become the biggest city of germany?` (Hamburg has never been the biggest city in Germany.) ",
                "finance": "-`how much is the worst performing stock, amazon?` (Amazon is one of the best performing stocks in the market.) ",
                "music": " - `how long was phil rudd the drummer for the band van halen?` (Phil Rudd was the drummer for AC/DC, and Alex Van Halen has been the primary drummer for Van Halen.) ",
                "movie": " - `when was \"soul\" released on hulu?` (The movie \"Soul\" was not released on Hulu. Instead, it was released on Disney+.)`",
                "sports": "- `what's the latest score update for OKC's game today?` (There is no game for OKC today)"
            }
        elif icl == 2:
            self.invalid_questions_examples  = {
                "finance": "-`when did hamburg become the biggest city of germany?` (Hamburg has never been the biggest city in Germany.) -`did taylor swifts debut album fearless launched in 2008 in us?` ( Taylor Swift's debut album was \"Taylor Swift\" (2006). \"Fearless\" was her second album, released in 2008.)",
                "movie": "-`how much is the worst performing stock, amazon?` (Amazon is one of the best performing stocks in the market.) -`what are the months where montana provide a ubi program?` (Montana does not have a UBI program)",
                "sports": " - `how long was phil rudd the drummer for the band van halen?` (Phil Rudd was the drummer for AC/DC, and Alex Van Halen has been the primary drummer for Van Halen.) - `what was the name of justin bieber's album last year?` (Justin Bieber did not release an album last year.)",
                "open": " - `when was \"soul\" released on hulu?` (The movie \"Soul\" was not released on Hulu. Instead, it was released on Disney+.) - `what year did the simpsons stop airing?` (\"The Simpsons\" is an ongoing series that has been continuously airing new episodes for over three decades.)",
                "music": "- `what's the latest score update for OKC's game today?` (There is no game for OKC today)- `how many times has curry won the nba dunk contest?` (Steph Curry has never participated in the NBA dunk contest) "
            }
        elif icl == 3:
            self.invalid_questions_examples  = {
                "open": "-`when did hamburg become the biggest city of germany?` (Hamburg has never been the biggest city in Germany.) -`did taylor swifts debut album fearless launched in 2008 in us?` ( Taylor Swift's debut album was \"Taylor Swift\" (2006). \"Fearless\" was her second album, released in 2008.) -`by how many votes did hillary clinton win the election in 2016?(Hillary Clinton didn't win the 2016 election; she lost to Donald Trump.)",
                "finance": "-`how much is the worst performing stock, amazon?` (Amazon is one of the best performing stocks in the market.) -`what are the months where montana provide a ubi program?` (Montana does not have a UBI program)  -`which five companies in the dow jones have a gross margin of less than 5%?(It assumes there are five Dow Jones companies with gross margins under 5%, which is unlikely.)",
                "music": " - `how long was phil rudd the drummer for the band van halen?` (Phil Rudd was the drummer for AC/DC, and Alex Van Halen has been the primary drummer for Van Halen.) - `what was the name of justin bieber's album last year?` (Justin Bieber did not release an album last year.) -`what day did beyonce die?(Beyoncé is alive, so the question is invalid.)",
                "movie": " - `when was \"soul\" released on hulu?` (The movie \"Soul\" was not released on Hulu. Instead, it was released on Disney+.) - `what year did the simpsons stop airing?` (\"The Simpsons\" is an ongoing series that has been continuously airing new episodes for over three decades.) -`was the lion king the highest-grossing film of all time when it was released in 1997?(The Lion King was released in 1994, not 1997.)",
                "sports": "- `what's the latest score update for OKC's game today?` (There is no game for OKC today)- `how many times has curry won the nba dunk contest?` (Steph Curry has never participated in the NBA dunk contest) -`how many wta 500 titles has coco gauff won in 2024?(As of 2024, Coco Gauff has not won any WTA 500 titles.)"
            }
        self.chunk_extractor = ChunkExtractor()


        

        if prompt_type == "base":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY)],
                )
            elif knowledge_source == "all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ALL)],
                )
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser()
        elif prompt_type == "cot":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY_COT)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY_COT)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY_COT)],
                )
            elif knowledge_source == "all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB_COT)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG_COT)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT)],
                )
            else:
                raise ValueError("Invalid knowledge source for COT prompt type")
        elif prompt_type =="cot_icl":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY_COT_ICL)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY_COT_ICL)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY_COT_ICL)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB_COT_ICL)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG_COT_ICL)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT_ICL)],
                )
            else:
                raise ValueError("Invalid knowledge source for COT prompt type")
        elif prompt_type == "baseline":
                if knowledge_source == "llm_web":
                    self.prompt_template = ChatPromptTemplate.from_messages(
                        [("system", BASELINE_SYSTEM_PROMPT), ("user", BASELINE_WEB)],
                    )
                elif knowledge_source == "llm_all":
                    self.prompt_template = ChatPromptTemplate.from_messages(
                        [("system", BASELINE_SYSTEM_PROMPT), ("user", BASELINE_ALL)],
                    )
                elif knowledge_source == "llm_kg":
                    self.prompt_template = ChatPromptTemplate.from_messages(
                        [("system", BASELINE_SYSTEM_PROMPT), ("user", BASELINE_KG)],
                    )
        if prompt_type == "cot" or prompt_type == "cot_icl":
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser() | self.get_final_answer
        else:
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser() 
        self.NER_prompt_template = ChatPromptTemplate.from_messages(
            [("system", Entity_Extract_TEMPLATE), ("user", NER_USER)],
        )
        self.get_entity_chain = self.NER_prompt_template | chat_model | StrOutputParser()
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3.1-8B-Instruct")

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.api = MockAPI(self.chat_model, self.terminators, self.tokenizer)
        self.classify_model_name = "Meta-Llama-3.1-8B-Instruct"
        self.use_domain = True
        if self.use_domain:
            domain_router_model_name = 'models/router/domain'
            self.domain_classes = ["finance", "music", "movie", "sports", "open"]
            
            
            self.router_tokenizer = AutoTokenizer.from_pretrained(self.classify_model_name)
            
            self.domain_router_model = AutoModelForSequenceClassification.from_pretrained(
                self.classify_model_name,
                device_map="cpu",
                num_labels=len(self.domain_classes),
                torch_dtype=torch.bfloat16,
            )
            self.domain_router_model = PeftModel.from_pretrained(self.domain_router_model, domain_router_model_name, adapter_name="domain_router")

            if self.router_tokenizer.pad_token is None:
                self.router_tokenizer.pad_token = self.router_tokenizer.eos_token
            if self.domain_router_model.config.pad_token_id is None:
                self.domain_router_model.config.pad_token_id = self.router_tokenizer.pad_token_id
        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "models/retrieve/embedding_models/all-MiniLM-L6-v2",
            device=torch.device(
                "cuda:4" if torch.cuda.is_available() else "cpu"
            ),
        )

    
    def get_final_answer(self, text):
            # 找到标志字符串的位置
        marker = "### Final Answer"
        marker_index = text.find(marker)
        # print("mark:", marker_index)
        if marker_index == -1:
            # 如果没有找到标志字符串，返回空字符串
            return "i don't know"
        
        # 获取标志字符串后面的内容
        content_start_index = marker_index + len(marker)
        final_answer_content = text[content_start_index:].strip()
        
        return final_answer_content

    def retrieve(self, input):
        query = input["query"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        return self.retriever.retrieve(query, interaction_id, search_results)
    
    def retrieve_baseline(self, input, max_chunks=3, max_chunk_length=200):
        # 从输入中获取 query、interaction_id 和 search_results
        query = input["query"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
    
        # 1. 提取 chunks，并限制每个 chunk 的长度
        batch_interaction_ids = [interaction_id]
        batch_search_results = [search_results]
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(batch_interaction_ids, batch_search_results)
    
        # 限制每个 chunk 的长度
        chunks = [chunk[:max_chunk_length] for chunk in chunks]
    
        # 2. 计算 query 和 chunks 的嵌入
        query_embedding = self.calculate_embeddings([query])[0]  # 因为这里只有一个 query，所以取第一个元素
        chunk_embeddings = self.calculate_embeddings(chunks)
    
        # 3. 计算余弦相似度
        cosine_scores = (chunk_embeddings * query_embedding).sum(1)
    
        # 4. 根据相似度排序，选择 top-N 的结果
        top_indices = (-cosine_scores).argsort()[:max_chunks]  # 只保留最相关的 N 个 chunks
        retrieval_results = [chunks[i] for i in top_indices]
    
        # 5. 返回检索到的结果
        return retrieval_results
    def extract_json_objects(self, text, decoder=JSONDecoder()):
        """Find the first JSON object in text, and return the decoded JSON data."""
        pos = 0
        while True:
            match = text.find("{", pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                return result  # 返回找到的第一个 JSON 对象
            except ValueError:
                pos = match + 1
        return None  # 如果没有找到 JSON 对象，返回 None
    def extract_entity(self, query, query_time):
        # 格式化单个 query 和 query_time 的提示词
        # formatted_prompt = self.format_prompt_for_entity_extraction(query, query_time)

        # 生成单个响应
        # response = self.llm.generate(
        #     [formatted_prompt],
        #     vllm.SamplingParams(
        #         n=1,  # 每个提示生成的输出序列数
        #         top_p=0.9,  # 控制选择的概率
        #         temperature=0.1,  # 生成过程中的随机性
        #         skip_special_tokens=True,  # 是否跳过输出中的特殊字符
        #         max_tokens=4096,  # 每个生成序列的最大token数
        #     ),
        #     use_tqdm=False  # 在本地开发时可设置为True以显示进度
        # )[0]  # 获取生成的第一个响应

        # # 解析响应以提取实体
        # res = response.outputs[0].text
        res = self.get_entity_chain.invoke({"query": query, "query_time": query_time})
        try:
            res = json.loads(res)
        except:
            res = self.extract_json_objects(res)
        # print("res:", res)
        # print(type(res))
        

        return res

    def get_kg_results(self, entity):
        if entity is None:
            return ""
        # print("entity:", entity)
        # print("type_entity:", type(entity))

        # Initialize the API
        api = CRAG(server=os.getenv("CRAG_MOCK_API_URL", "http://localhost:8001"))
        kg_results = []
        res = ""

        # Check the domain and extract information accordingly
        if "domain" in entity.keys():
            domain = entity["domain"]

            if domain in ["encyclopedia", "other"]:
                if "main_entity" in entity.keys():
                    try:
                        top_entity_name = api.open_search_entity_by_name(entity["main_entity"])["result"][0]
                        res = api.open_get_entity(top_entity_name)["result"]
                        kg_results.append({top_entity_name: res})
                    except Exception as e:
                        logger.warning(f"Error in open_get_entity: {e}")
                        pass

            elif domain == "movie":
                if "movie_name" in entity.keys() and entity["movie_name"] is not None:
                    if isinstance(entity["movie_name"], str):
                        movie_names = entity["movie_name"].split(",")
                    else:
                        movie_names = entity["movie_name"]
                    for movie_name in movie_names:
                        try:
                            res = api.movie_get_movie_info(movie_name)["result"][0]
                            res = res[entity["movie_aspect"]]
                            kg_results.append({movie_name + "_" + entity["movie_aspect"]: res})
                        except Exception as e:
                            logger.warning(f"Error in movie_get_movie_info: {e}")
                            pass

                if "person" in entity.keys() and entity["person"] is not None:
                    if isinstance(entity["person"], str):
                        person_list = entity["person"].split(",")
                    else:
                        person_list = entity["person"]
                    for person in person_list:
                        try:
                            res = api.movie_get_person_info(person)["result"][0]
                            aspect = entity["person_aspect"]
                            if aspect in ["oscar_awards", "birthday"]:
                                res = res[aspect]
                                kg_results.append({person + "_" + aspect: res})
                            if aspect in ["acted_movies", "directed_movies"]:
                                movie_info = []
                                for movie_id in res[aspect]:
                                    movie_info.append(api.movie_get_movie_info_by_id(movie_id))
                                kg_results.append({person + "_" + aspect: movie_info})
                        except Exception as e:
                            logger.warning(f"Error in movie_get_person_info: {e}")
                            pass

                if "year" in entity.keys() and entity["year"] is not None:
                    if isinstance(entity["year"], str) or isinstance(entity["year"], int):
                        years = str(entity["year"]).split(",")
                    else:
                        years = entity["year"]
                    for year in years:
                        try:
                            res = api.movie_get_year_info(year)["result"]
                            all_movies = []
                            oscar_movies = []
                            for movie_id in res["movie_list"]:
                                all_movies.append(api.movie_get_movie_info_by_id(movie_id)["result"]["title"])
                            for movie_id in res["oscar_awards"]:
                                oscar_movies.append(api.movie_get_movie_info_by_id(movie_id)["result"]["title"])   
                            kg_results.append({year + "_all_movies": all_movies})
                            kg_results.append({year + "_oscar_movies": oscar_movies})
                        except Exception as e:
                            logger.warning(f"Error in movie_get_year_info: {e}")
                            pass

        # Combine results into a single string format and return
        return "<DOC>\n".join([str(res) for res in kg_results]) [:10000]if len(kg_results) > 0 else ""

 



    
    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    
    def generate_answer(self, query, interaction_id, search_results, kg_info, query_time, domain):
        contexts = []
        # print("kg_info:", kg_info) #not empty
        if self.prompt_type == "cot_icl":
            if self.knowledge_source in ["web", "llm_web", "all", "llm_all"]:
                # Retrieve references from Web
                contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results, "invalid_questions_examples":self.invalid_questions_examples[domain]})
                web_info = self.get_reference(contexts)
            if self.knowledge_source == "llm":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "invalid_questions_examples":self.invalid_questions_examples[domain]})
            elif self.knowledge_source in ["web", "llm_web"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info, "invalid_questions_examples":self.invalid_questions_examples[domain]})
            elif self.knowledge_source in ["kg", "llm_kg"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": kg_info, "invalid_questions_examples":self.invalid_questions_examples[domain]})
            elif self.knowledge_source in ["all", "llm_all"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info, "invalid_questions_examples":self.invalid_questions_examples[domain]})
            # print("invalid_questions_examples:", self.invalid_questions_examples[domain])
        elif self.prompt_type == "cot":
            if self.knowledge_source in ["web", "llm_web", "all", "llm_all"]:
                # Retrieve references from Web
                contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                web_info = self.get_reference(contexts)
            if self.knowledge_source == "llm":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time})
            elif self.knowledge_source in ["web", "llm_web"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info})
            elif self.knowledge_source in ["kg", "llm_kg"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": kg_info})
            elif self.knowledge_source in ["all", "llm_all"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info})
            # print("invalid_questions_examples:", self.invalid_questions_examples[domain])
        elif self.prompt_type == "baseline":
            if self.knowledge_source in ["web", "llm_web", "all", "llm_all"]:
                # Retrieve references from Web
                # contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                contexts = self.retrieve_baseline({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                web_info = self.get_reference(contexts)
            if self.knowledge_source in ["llm_all","llm_kg"]:
                entities = self.extract_entity( query,  query_time)
                kg_info = self.get_kg_results(entities)
            if self.knowledge_source == "llm_web":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info})
            elif self.knowledge_source == "llm_all":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info})
            elif self.knowledge_source == "llm_kg":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "kg_infos": kg_info})
        
        else:
            if self.knowledge_source in ["web", "llm_web", "all", "llm_all"]:
                # Retrieve references from Web
                contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                web_info = self.get_reference(contexts)
            if self.knowledge_source == "llm":
                response = self.rag_chain.invoke({"query": query, "query_time": query_time})
            elif self.knowledge_source in ["web", "llm_web"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info})
            elif self.knowledge_source in ["kg", "llm_kg"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": kg_info})
            elif self.knowledge_source in ["all", "llm_all"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info})
            # print("contexts:", contexts)
        print("response:", response)
        return response, contexts
    
    def generate_answer_(self, input):
        query = input["query"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        kg_info = input["kg_info"]
        query_time = input["query_time"]
        domain = input["domain"]
        # print("kg_info:", kg_info)
        return self.generate_answer(query, interaction_id, search_results, kg_info, query_time, domain)

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        # kg_infos = batch["kg_info"]
        # kg_infos = [""] * len(queries)
        domains = [""] * len(queries)
        if self.use_domain:
            domains = self.classify_question_domain(queries)
            domains = [self.domain_classes[domain] for domain in domains]
            kg_infos = self.api.get_kg_info(queries, query_times, domains)
        else:
            kg_infos = [""] * len(queries)
        # print("kg_infos:", kg_infos)

        outputs = RunnableLambda(self.generate_answer_).batch([{"query": query, "interaction_id": interaction_id, "search_results": search_results, "kg_info": kg_info, "query_time": query_time, "domain":domain} for query, interaction_id, search_results, kg_info, query_time,domain in zip(queries, batch_interaction_ids, batch_search_results, kg_infos, query_times, domains)])
        answers = [output[0] for output in outputs]
        batch_contexts = [output[1] for output in outputs]
        # print("batch_contexts:", batch_contexts)
        return answers, batch_contexts
    
    def get_reference(self, retrieval_results):
        references = ""
        if len(retrieval_results) > 1:
            for _snippet_idx, snippet in enumerate(retrieval_results):
                references += "<DOC>\n"
                references += f"{snippet.strip()}\n"
                references += "</DOC>\n\n"
        elif len(retrieval_results) == 1 and len(retrieval_results[0]) > 0:
            references = retrieval_results[0]
        else:
            references = "No References"
        return references
    def classify_question_domain(self, queries: str) -> int:
        """
        Classify the question type based on the provided query.

        Parameters:
        - query (str): The user's question.

        Returns:
        - str: The predicted question type.

        This method processes the user's question using a pre-trained router model to classify it into
        one of the following categories: simple, simple_w_condition, comparison, aggregation, set, post-processing, multi-hop.
        """
        # Tokenize the query and generate a classification prediction.
        inputs = self.router_tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

        # Send all tokenized queries to the model in one batch.
        outputs = self.domain_router_model(**inputs)
        logits = outputs.logits

        # Extract the predicted class indices for each query in the batch.
        predicted_class_indices = logits.argmax(dim=1).tolist()

        return predicted_class_indices

class RAGModel_2Stage:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, chat_model, retriever, knowledge_source, prompt_type="base"):
        self.chat_model = chat_model
        self.retriever = retriever
        self.knowledge_source = knowledge_source
        SYSTEM_PROMPT = "You are a helpful assistant."
        
        self.llm_prompt_template = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", LLM_ONLY)],
        )

        if prompt_type == "base":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY)],
                )
            elif knowledge_source == "all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ALL)],
                )
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser()
        elif prompt_type == "cot":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY_COT)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY_COT)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY_COT)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB_COT)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG_COT)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT)],
                )
            else:
                raise ValueError("Invalid knowledge source for COT prompt type")
        elif prompt_type =="cot_icl":
            if knowledge_source == "llm":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_ONLY_COT_ICL)],
                )
            elif knowledge_source == "web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY_COT_ICL)],
                )
            elif knowledge_source == "kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY_COT_ICL)],
                )
            elif knowledge_source == "llm_web":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_WEB_COT_ICL)],
                )
            elif knowledge_source == "llm_kg":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", LLM_KG_COT_ICL)],
                )
            elif knowledge_source == "llm_all":
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT_ICL)],
                )
            else:
                raise ValueError("Invalid knowledge source for COT prompt type")
        self.llm_chain = self.llm_prompt_template | chat_model | StrOutputParser()
        if prompt_type == "cot":
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser() | self.get_final_answer
        else:
            self.rag_chain = self.prompt_template | chat_model | StrOutputParser()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3.1-8B-Instruct")

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.api = MockAPI(self.chat_model, self.terminators, self.tokenizer)
        self.classify_model_name = "Meta-Llama-3.1-8B-Instruct"
        self.use_domain = True
        if self.use_domain:
            domain_router_model_name = 'models/router/domain'
            self.domain_classes = ["finance", "music", "movie", "sports", "open"]
            
            
            self.router_tokenizer = AutoTokenizer.from_pretrained(self.classify_model_name)
            
            self.domain_router_model = AutoModelForSequenceClassification.from_pretrained(
                self.classify_model_name,
                device_map="cpu",
                num_labels=len(self.domain_classes),
                # quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
            self.domain_router_model = PeftModel.from_pretrained(self.domain_router_model, domain_router_model_name, adapter_name="domain_router")

            if self.router_tokenizer.pad_token is None:
                self.router_tokenizer.pad_token = self.router_tokenizer.eos_token
            if self.domain_router_model.config.pad_token_id is None:
                self.domain_router_model.config.pad_token_id = self.router_tokenizer.pad_token_id
    def get_final_answer(self, text):
            # 找到标志字符串的位置
        marker = "### Final Answer"
        marker_index = text.find(marker)
        # print("mark:", marker_index)
        if marker_index == -1:
            # 如果没有找到标志字符串，返回空字符串
            return "i don't know"
        
        # 获取标志字符串后面的内容
        content_start_index = marker_index + len(marker)
        final_answer_content = text[content_start_index:].strip()
        
        return final_answer_content           
    def retrieve(self, input):
        query = input["query"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        return self.retriever.retrieve(query, interaction_id, search_results)
    
    def generate_answer(self, query, interaction_id, search_results, kg_info, query_time):
        response = self.llm_chain.invoke({"query": query, "query_time": query_time})
        contexts = []
        if "i don't know" in response.lower():
            if self.knowledge_source in ["web", "llm_web", "all", "llm_all"]:
                # Retrieve references from Web
                contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                web_info = self.get_reference(contexts)
            if self.knowledge_source in ["web", "llm_web"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info})
            elif self.knowledge_source in ["kg", "llm_kg"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "references": kg_info})
            elif self.knowledge_source in ["all", "llm_all"]:
                response = self.rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info})
        return response, contexts
    
    def generate_answer_(self, input):
        query = input["query"]
        query_time = input["query_time"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        kg_info = input["kg_info"]
        return self.generate_answer(query, interaction_id, search_results, kg_info, query_time)

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        # kg_infos = batch["kg_info"]
        # kg_infos = [""] * len(queries)
        if self.use_domain:
            domains = self.classify_question_domain(queries)
            domains = [self.domain_classes[domain] for domain in domains]
            kg_infos = self.api.get_kg_info(queries, query_times, domains)
        else:
            kg_infos = [""] * len(queries)
        # print("kg_infos:", kg_infos)

        outputs = RunnableLambda(self.generate_answer_).batch([{"query": query, "interaction_id": interaction_id, "search_results": search_results, "kg_info": kg_info, "query_time": query_time} for query, interaction_id, search_results, kg_info, query_time in zip(queries, batch_interaction_ids, batch_search_results, kg_infos, query_times)])
        answers = [output[0] for output in outputs]
        batch_contexts = [output[1] for output in outputs]
        # print("batch_contexts:", batch_contexts)
        return answers, batch_contexts
    
    def get_reference(self, retrieval_results):
        references = ""
        if len(retrieval_results) > 1:
            for _snippet_idx, snippet in enumerate(retrieval_results):
                references += "<DOC>\n"
                references += f"{snippet.strip()}\n"
                references += "</DOC>\n\n"
        elif len(retrieval_results) == 1 and len(retrieval_results[0]) > 0:
            references = retrieval_results[0]
        else:
            references = "No References"
        return references
    def classify_question_domain(self, queries: str) -> int:
        """
        Classify the question type based on the provided query.

        Parameters:
        - query (str): The user's question.

        Returns:
        - str: The predicted question type.

        This method processes the user's question using a pre-trained router model to classify it into
        one of the following categories: simple, simple_w_condition, comparison, aggregation, set, post-processing, multi-hop.
        """
        # Tokenize the query and generate a classification prediction.
        inputs = self.router_tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

        # Send all tokenized queries to the model in one batch.
        outputs = self.domain_router_model(**inputs)
        logits = outputs.logits

        # Extract the predicted class indices for each query in the batch.
        predicted_class_indices = logits.argmax(dim=1).tolist()

        return predicted_class_indices
    
class RAGModel_3Stage:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, chat_model, retriever, knowledge_source):
        self.chat_model = chat_model
        self.retriever = retriever
        self.knowledge_source = knowledge_source
        SYSTEM_PROMPT = "You are a helpful assistant."
        self.llm_prompt_template = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", LLM_ONLY_COT)],
        )
        self.path_selection_prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", PATH_SELECTION)],
        )
        self.path_chain = self.path_selection_prompt | chat_model | StrOutputParser() | self.path_select
        self.llm_chain = self.llm_prompt_template | chat_model | StrOutputParser()  | self.get_final_answer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3.1-8B-Instruct")

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.api = MockAPI(self.chat_model, self.terminators, self.tokenizer)
        self.classify_model_name = "Meta-Llama-3.1-8B-Instruct"
        self.use_domain = True
        if self.use_domain:
            domain_router_model_name = 'models/router/domain'
            self.domain_classes = ["finance", "music", "movie", "sports", "open"]
            
            
            self.router_tokenizer = AutoTokenizer.from_pretrained(self.classify_model_name)
            
            self.domain_router_model = AutoModelForSequenceClassification.from_pretrained(
                self.classify_model_name,
                device_map="cpu",
                num_labels=len(self.domain_classes),
                # quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
            self.domain_router_model = PeftModel.from_pretrained(self.domain_router_model, domain_router_model_name, adapter_name="domain_router")

            if self.router_tokenizer.pad_token is None:
                self.router_tokenizer.pad_token = self.router_tokenizer.eos_token
            if self.domain_router_model.config.pad_token_id is None:
                self.domain_router_model.config.pad_token_id = self.router_tokenizer.pad_token_id
    def get_final_answer(self, text):
            # 找到标志字符串的位置
        marker = "### Final Answer"
        marker_index = text.find(marker)
        # print("mark:", marker_index)
        if marker_index == -1:
            # 如果没有找到标志字符串，返回空字符串
            # print("i don't know1111111111111111111111111111111111")
            return "i don't know"
        
        # 获取标志字符串后面的内容
        content_start_index = marker_index + len(marker)
        final_answer_content = text[content_start_index:].strip()    
        return final_answer_content
            
    def path_select(self, text):
        if len(text) == 1:
            if text == "a":
                return "web"
            elif text == "b":
                return "kg"
            elif text == "c":
                return "all"
            else:
                return "wrong"
        elif len(text) > 1:
            if text[:2] == "a " or text[:2] == "a)":
                return "web"
            elif text[:2] == "b " or text[:2] == "b)":
                return "kg"
            elif text[:2] == "c " or text[:2] == "c)":
                return "all"
            else:
                return "wrong"

    def retrieve(self, input):
        query = input["query"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        return self.retriever.retrieve(query, interaction_id, search_results)
    
    def generate_answer(self, query, interaction_id, search_results, kg_info, query_time):
        response = self.llm_chain.invoke({"query": query, "query_time": query_time})
        print("response:", response)
        # print("response_type:", type(response))
        contexts = []
        SYSTEM_PROMPT = "You are a helpful assistant."
        if response == None or "i don't know" in response.lower():
            knowledge_source = self.path_chain.invoke({"query": query})
            print("knowledge_source:", knowledge_source)
            if knowledge_source in ["web", "all"]:
                # Retrieve references from Web
                contexts = self.retrieve({"query": query, "interaction_id": interaction_id, "search_results": search_results})
                web_info = self.get_reference(contexts)
            if knowledge_source in ["web"]:
                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", WEB_ONLY_COT)],
                )
                rag_chain = prompt_template | self.chat_model | StrOutputParser() | self.get_final_answer
                response = rag_chain.invoke({"query": query, "query_time": query_time, "references": web_info})
            elif knowledge_source in ["kg"]:
                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", KG_ONLY_COT)],
                )
                rag_chain = prompt_template | self.chat_model | StrOutputParser() | self.get_final_answer
                response = rag_chain.invoke({"query": query, "query_time": query_time, "references": kg_info})
            elif knowledge_source in ["all"]:
                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("user", ALL_COT)],
                )
                rag_chain = prompt_template | self.chat_model | StrOutputParser() | self.get_final_answer
                response = rag_chain.invoke({"query": query, "query_time": query_time, "web_infos": web_info, "kg_infos": kg_info})
            else:
                response = "I don't know"
            print("response1:", response)
            # print("response_type:", type(response))
            # print("contexts:", contexts)
        return response, contexts
    
    def generate_answer_(self, input):
        query = input["query"]
        query_time = input["query_time"]
        interaction_id = input["interaction_id"]
        search_results = input["search_results"]
        kg_info = input["kg_info"]
        return self.generate_answer(query, interaction_id, search_results, kg_info, query_time)

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        # kg_infos = batch["kg_info"]
        # kg_infos = [""] * len(queries)
        if self.use_domain:
            domains = self.classify_question_domain(queries)
            domains = [self.domain_classes[domain] for domain in domains]
            kg_infos = self.api.get_kg_info(queries, query_times, domains)
        else:
            kg_infos = [""] * len(queries)
        # print("kg_infos:", kg_infos)

        outputs = RunnableLambda(self.generate_answer_).batch([{"query": query, "interaction_id": interaction_id, "search_results": search_results, "kg_info": kg_info, "query_time": query_time} for query, interaction_id, search_results, kg_info, query_time in zip(queries, batch_interaction_ids, batch_search_results, kg_infos, query_times)])
        answers = [output[0] for output in outputs]
        batch_contexts = [output[1] for output in outputs]
        # print("batch_contexts:", batch_contexts)
        return answers, batch_contexts
    
    def get_reference(self, retrieval_results):
        references = ""
        if len(retrieval_results) > 1:
            for _snippet_idx, snippet in enumerate(retrieval_results):
                references += "<DOC>\n"
                references += f"{snippet.strip()}\n"
                references += "</DOC>\n\n"
        elif len(retrieval_results) == 1 and len(retrieval_results[0]) > 0:
            references = retrieval_results[0]
        else:
            references = "No References"
        return references
    
    def classify_question_domain(self, queries: str) -> int:
        """
        Classify the question type based on the provided query.

        Parameters:
        - query (str): The user's question.

        Returns:
        - str: The predicted question type.

        This method processes the user's question using a pre-trained router model to classify it into
        one of the following categories: simple, simple_w_condition, comparison, aggregation, set, post-processing, multi-hop.
        """
        # Tokenize the query and generate a classification prediction.
        inputs = self.router_tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

        # Send all tokenized queries to the model in one batch.
        outputs = self.domain_router_model(**inputs)
        logits = outputs.logits

        # Extract the predicted class indices for each query in the batch.
        predicted_class_indices = logits.argmax(dim=1).tolist()

        return predicted_class_indices

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

#### CONFIG PARAMETERS ---


# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids