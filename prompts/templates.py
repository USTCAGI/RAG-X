#!/usr/bin/env python3

INSTRUCTIONS = """
# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# Output: 
Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".
"""

IN_CONTEXT_EXAMPLES = """
# Examples:
Question: how many seconds is 3 minutes 15 seconds?
Ground truth: ["195 seconds"]
Prediction: 3 minutes 15 seconds is 195 seconds.
Accuracy: True

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
Accuracy: False

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I am sorry I don't know.
Accuracy: False
"""

PATH_SELECTION = """You are an intelligent assistant tasked with selecting the most appropriate data source(s) for answering user queries. You have access to two types of data sources:

1. Web Pages: Obtained through search engines, providing rich and comprehensive information but may contain outdated or misleading information for time-sensitive queries.

2. Mock APIs: Real-time APIs offering current information, less comprehensive than web pages but highly reliable for time-sensitive data.

Based on the user's query, you must choose the most suitable data source(s) from these options:
a) Web Pages only
b) Mock APIs only
c) Both Web Pages and Mock APIs

Consider the following factors when making your decision:
- The nature of the query (static information vs. time-sensitive data)
- The need for comprehensive information
- The importance of up-to-date information
- The potential for complementary information from both sources

Your response should be only one of the following options: a, b, or c.
Do not provide any explanation or additional information.

### Query
{query}
### Option
"""

LLM_ONLY_ = """Please answer the following question. Your answer should be short and concise.
Current Time: {query_time}

Note: 
- For your final answer, please use as few words as possible. 
- The user's question may contain factual errors, in which case you MUST reply `invalid question`.
- If you don't know the answer, you MUST respond with `I don't know`.

### Question
{query}

### Answer
"""

LLM_ONLY = """You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### Answer
"""

LLM_WEB = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your answer should be short and concise, using as few words as possible.
### Query Time
{query_time}
### References
{references}
### Answer
"""



LLM_KG = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Answer
"""

LLM_ALL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
#### Web Infos
{web_infos}
#### KG Infos
{kg_infos}
### Answer
"""

WEB_ONLY = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Answer
"""

KG_ONLY = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Answer
"""

ALL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your answer should be short and concise, using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
#### Web Infos
{web_infos}
#### KG Infos
{kg_infos}
### Answer
"""

LLM_ONLY_COT = """You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### Thought
"""

LLM_WEB_COT = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""

LLM_KG_COT = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""

ALL_COT = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
#### Web Infos
{web_infos}
#### KG Infos
{kg_infos}
### Thought
"""

WEB_ONLY_COT = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your final answer should be short and concise, using as few words as possible.
6. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""
KG_ONLY_COT = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your final answer should be short and concise, using as few words as possible.
6. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""



LLM_ONLY_COT_ICL = """You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### Thought
"""

LLM_WEB_COT_ICL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""

LLM_KG_COT_ICL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""

ALL_COT_ICL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question, please think step by step, then provide the final answer.
Please follow these guidelines when formulating your answer:
1. The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Your final answer should be short and concise, using as few words as possible.
4. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
#### Web Infos
{web_infos}
#### KG Infos
{kg_infos}
### Thought
"""

WEB_ONLY_COT_ICL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your final answer should be short and concise, using as few words as possible.
6. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""
KG_ONLY_COT_ICL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as `Query Time`. The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. - The user's question may contain factual errors, in which case you MUST reply `invalid question` Here are some examples of invalid questions:
    {invalid_questions_examples}
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. If the references do not contain the necessary information to answer the question, respond with `I don't know`.
4. Using only the refernces below and not prior knowledge, if there is no reference, respond with `I don't know`.
5. Your final answer should be short and concise, using as few words as possible.
6. Your output format needs to meet the requirements: First, start with `### Thought\n` and then output the thought process regarding the user's question. After you finish thinking, you MUST reply with the final answer on the last line, starting with `### Final Answer\n` and using as few words as possible.
### Question
{query}
### Query Time
{query_time}
### References
{references}
### Thought
"""


BASELINE_WEB = """
# References 
{references}

------

Using only the references listed above, answer the following question: 
Current Time: {query_time}
Question: {query}
"""

BASELINE_ALL = """
### References 
# Web
{web_infos}
# Knowledge Graph
{kg_infos}
------

Using only the references listed above, answer the following question: 
Current Time: {query_time}
Question: {query}
"""

BASELINE_KG = """
### References 
# Knowledge Graph
{kg_infos}
------

Using only the references listed above, answer the following question: 
Current Time: {query_time}
Question: {query}
"""
Entity_Extract_TEMPLATE = """
You are given a Query and Query Time. Do the following: 

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

For `encyclopedia` and `other` queries, these are possible keys:
-  `main_entity`: extract the main entity of the query. 

For `finance` queries, these are possible keys:
- `market_identifier`: stock identifiers including individual company names, stock symbols.
- `metric`: financial metrics that the query is asking about. This must be one of the following: `price`, `dividend`, `P/E ratio`, `EPS`, `marketCap`, and `other`.
- `datetime`: time frame that query asks about. When datetime is not explicitly mentioned, use `Query Time` as default. 

For `movie` queries, these are possible keys:
- `movie_name`: name of the movie
- `movie_aspect`: if the query is about a movie, which movie aspect the query asks. This must be one of the following: `budget`, `genres`, `original_language`, `original_title`, `release_date`, `revenue`, `title`, `cast`, `crew`, `rating`, `length`.
- `person`: person name related to moves
- `person_aspect`: if the query is about a person, which person aspect the query asks. This must be one of the following: `acted_movies`, `directed_movies`, `oscar_awards`, `birthday`.
- `year`: if the query is about movies released in a specific year, extract the year

For `music` queries, these are possible keys:
- `artist_name`: name of the artist
- `artist_aspect`: if the query is about an artist, extract the aspect of the artist. This must be one of the following: `member`, `birth place`, `birth date`, `lifespan`, `artist work`, `grammy award count`, `grammy award date`.
- `song_name`: name of the song
- `song_aspect`: if the query is about a song, extract the aspect of the song. This must be one of the following: `auther`, `grammy award count`, `release country`, `release date`.

For `sports` queries, these are possible keys:
- `sport_type`: one of `basketball`, `soccer`, `other`
- `tournament`: such as NBA, World Cup, Olympic.
- `team`: teams that user interested in.
- `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

Return the results in a FLAT json. 

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
"""

NER_USER = """
Query: {query}
Query Time: {query_time}
"""