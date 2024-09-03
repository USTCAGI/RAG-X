import re
import vllm

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---
NER_OPEN_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

please identify and list all the named entities present in the following question instead answering it, categorizing them appropriately (e.g., person, location, orgnization, product, event and so on)Your answer should be short and concise in 50 words.
Format your response as follows: For each entity, provide the name followed by its category in parentheses.  Categories include person, location, orgnization, product, event and so on. Ensure that your response is clearly structured and easy to read."
Expected Output Format:
a name of a person in the sentence(person)
a name of a place  in the sentence(location)
a name of a orgnization in the sentence(orgnization)
a name of a product in the sentence(product)
a name of a event in the sentence(event)
every entity should be in a new line and be in the format of "entity_name (entity_category)"
### Question
{query}


<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
NER_MUSIC_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

please identify and list all the named entities present in the following question about music instead answering it, categorizing them appropriately (e.g., persons, song, band) Your answer should be short and concise in 50 words.
Format your response as follows: For each entity, provide the name followed by its category in parentheses. Categories include persons, songs and bands. Ensure that your response is clearly structured and easy to read."

Expected Output Format:

a name of a person in the sentence  (person)
a name of a song  in the sentence  (song)
a name of a band  in the sentence  (band)
every entity should be in a new line and be in the format of "entity_name (entity_category)"
### Question
{query}


<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
NER_FINANCE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

please identify and list all the named entities present in the following question about finance instead answering it, categorizing them appropriately (e.g., company, ticker) Your answer should be short and concise in 50 words.
Format your response as follows: For each entity, provide the name followed by its category in parentheses. Categories include company, and ticker. Ensure that your response is clearly structured and easy to read."
Expected Output Format:
a name of a company in the sentence  (company)
a name of a ticker  in the sentence  (ticker)
every entity should be in a new line and be in the format of "entity_name (entity_category)"
### Question
{query}


<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
NER_MOVIE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

please identify and list all the named entities present in the following question about movie instead answering it, categorizing them appropriately (e.g., persons,  movie) Your answer should be short and concise in 50 words.
Format your response as follows: For each entity, provide the name followed by its category in parentheses. Categories include persons, and movies. Ensure that your response is clearly structured and easy to read."

Expected Output Format:

a name of a person in the sentence  (person)
a name of a movie  in the sentence  (movie)
every entity should be in a new line and be in the format of "entity_name (entity_category)"
### Question
{query}


<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
NER_SPORTS_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

please identify and list all the named entities present in the following question about sports instead answering it, categorizing them appropriately (e.g., nba team, soccer team) Your answer should be short and concise in 50 words.
Format your response as follows: For each entity, provide the name followed by its category in parentheses. Categories include nba team, and soccer team. Ensure that your response is clearly structured and easy to read."

Expected Output Format:

a name of a nba team in the sentence  (nbateam)
a name of a soccer team  in the sentence  (soccerteam)
every entity should be in a new line and be in the format of "entity_name (entity_category)"
### Question
{query}


<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
LIMIT_ENTITY_WORDS = 5

import datetime
import pytz
import re

def parse_datetime(datetime_str):
    # 移除末尾的时区标识 "PT" 和它前面的空格
    datetime_str = datetime_str.replace(' PT', '')
    
    # 定义日期时间的格式
    date_format = "%m/%d/%Y, %H:%M:%S"

    # 使用strptime解析日期时间字符串
    dt = datetime.datetime.strptime(datetime_str, date_format)

    # 由于原始字符串包含时区 "PT"，我们可以将其解析为太平洋时区
    pacific_timezone = pytz.timezone('America/Los_Angeles')
    dt = pacific_timezone.localize(dt)

    return dt.date()  # 返回日期部分，去除时间

def find_date_from_text(current_date, text):
    # 正则表达式匹配各种日期描述
    pattern = r"(last (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|yesterday)"
    match = re.search(pattern, text)

    if not match:
        return None  # 如果没有匹配，返回 None

    # 匹配到的字符串
    date_str = match.group(0)

    # 返回相应的日期
    return calculate_date(current_date, date_str).strftime('%Y-%m-%d')

def calculate_date(current_date, date_str):
    today = parse_datetime(current_date)
    
    # 处理 "today" 和 "yesterday"
    if date_str == "today":
        return today
    elif date_str == "yesterday":
        return today - datetime.timedelta(days=1)

    # 处理 "last Monday" 到 "last Sunday"
    weekdays = {
        "last monday": 0,
        "last tuesday": 1,
        "last wednesday": 2,
        "last thursday": 3,
        "last friday": 4,
        "last saturday": 5,
        "last sunday": 6
    }

    target_weekday = weekdays[date_str]
    current_weekday = today.weekday()

    # 计算天数差
    days_difference = current_weekday - target_weekday

    # 如果days_difference为非负，表明"上一个"目标星期几在当前日期之前或当天，需回退一周
    if days_difference <= 7:
        days_difference += 7

    return today - datetime.timedelta(days=days_difference)

# 输入两个参数，字符串类型的query time和文本，输出文本中涉及的时间
#示例
# datetime_string = "03/05/2024, 23:18:31 PT"
# text = "We will meet last Friday"
# print(find_date_from_text(datetime_string, text))  # 输出上一个星期五的日期
# text = "We will meet yesterday"
# print(find_date_from_text(datetime_string, text))  # 输出昨天的日期

class NER:
    def __init__(self, llm):
        self.llm = llm

    def movie_process_text(self, input_text):
        # 使用正则表达式找到从行的开始到 (movie) 或 (person) 的整个片段
        pattern = re.compile(r'^(.*?)(\((movie|person)\))', re.MULTILINE)
        matches = pattern.finditer(input_text)

        results = {
            'movie': set(),
            'person': set()
        }
        # 检查每个匹配，根据 (movie) 或 (person) 保存到相应的变量中
        for match in matches:
            text = match.group(1).strip()
            # 删除所有非字母字符开头的部分
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            # words = text.split()
            # if len(words) > LIMIT_ENTITY_WORDS:
            #     continue  # 如果单词数超过5个，跳过此次记录
            if 'none' in text.lower():
                continue
            if match.group(3) == 'movie':
                results['movie'].add(text)
            elif match.group(3) == 'person':
                results['person'].add(text)
        # 将集合转换为列表
        results['movie'] = list(results['movie'])
        results['person'] = list(results['person'])

        return results

    def sports_process_text(self, input_text):
        # 使用正则表达式找到从行的开始到 (team) 或 (person) 的整个片段
        pattern = re.compile(r'^(.*?)(\((soccerteam|nbateam)\))', re.MULTILINE)
        matches = pattern.finditer(input_text)

        # results = {
        #     'soccerteam': set(),
        #     'nbateam': set()
        # }
        # 使用集合来存储结果，以避免重复
        results = {
            'soccerteam': set(),
            'nbateam': set()
        }

        # 检查每个匹配，根据 (team) 或 (person) 保存到相应的变量中
        for match in matches:
            text = match.group(1).strip()
            # 删除所有非字母字符开头的部分
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            # words = text.split()
            # if len(words) > LIMIT_ENTITY_WORDS:
            #     continue  # 如果单词数超过5个，跳过此次记录
            if 'none' in text.lower():
                continue
            if match.group(3) == 'soccerteam':
                results['soccerteam'].add(text)
            elif match.group(3) == 'nbateam':
                results['nbateam'].add(text)
            # 将集合转换为列表
        results['soccerteam'] = list(results['soccerteam'])
        results['nbateam'] = list(results['nbateam'])
        return results

    def music_process_text(self, input_text):
        # 使用正则表达式找到从行的开始到 (team) 或 (person) 的整个片段
        pattern = re.compile(r'^(.*?)(\((person|song|band)\))', re.MULTILINE)
        matches = pattern.finditer(input_text)
        results = {
            'person': set(),
            'song': set(),
            'band': set()
        }

        # 检查每个匹配，根据 (team) 或 (person) 保存到相应的变量中
        for match in matches:
            text = match.group(1).strip()
            # 删除所有非字母字符开头的部分
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            # words = text.split()
            # if len(words) > LIMIT_ENTITY_WORDS:
            #     continue  # 如果单词数超过5个，跳过此次记录
            if 'none' in text.lower():
                continue
            if match.group(3) == 'person':
                parts = text.split(',')
                for part in parts:
                    results['person'].add(part)
                # text_1.add(text)
            elif match.group(3) == 'song':
                parts = text.split(',')
                for part in parts:
                    results['song'].add(part)
                # text_2.add(text)
            elif match.group(3) == 'band':
                parts = text.split(',')
                for part in parts:
                    results['band'].add(part)
                # text_4.add(text)
        results['person'] = list(results['person'])
        results['song'] = list(results['song'])
        results['band'] = list(results['band'])

        return results

    def finance_process_text(self, input_text):
        # 使用正则表达式找到从行的开始到 (team) 或 (person) 的整个片段
        pattern = re.compile(r'^(.*?)(\((company|ticker)\))', re.MULTILINE)
        matches = pattern.finditer(input_text)

        results = {
            'company': set(),
            'ticker': set()
        }

        # 检查每个匹配，根据 (team) 或 (person) 保存到相应的变量中
        for match in matches:
            text = match.group(1).strip()
            # 删除所有非字母字符开头的部分
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            # words = text.split()
            # if len(words) > LIMIT_ENTITY_WORDS:
            #     continue  # 如果单词数超过5个，跳过此次记录
            if 'none' in text.lower():
                continue
            if match.group(3) == 'company':
                results['company'].add(text)
            elif match.group(3) == 'ticker':
                results['ticker'].add(text)
        results['company'] = list(results['company'])
        results['ticker'] = list(results['ticker'])
        return results


    def open_process_text(self, input_text):
        # 使用正则表达式找到从行的开始到 (team) 或 (person) 的整个片段
        pattern = re.compile(r'^(.*?)(\((\w+)\))', re.MULTILINE)
        matches = pattern.finditer(input_text)

        results = {
            'entity': set()
        }

        # 检查每个匹配，根据 (team) 或 (person) 保存到相应的变量中
        for match in matches:
            text = match.group(1).strip()
            # 删除所有非字母字符开头的部分
            text = re.sub(r'^[^a-zA-Z]+', '', text)
            # words = text.split()
            # if len(words) > LIMIT_ENTITY_WORDS:
            #     continue  # 如果单词数超过5个，跳过此次记录

            if 'none' in text.lower():
                continue
            parts = text.split(',')
            for part in parts:
                results['entity'].add(part)
        results['entity'] = list(results['entity'])
        return results
    
    def generate_answer(self, query: str, domain: str, query_time: str) -> dict:
        references = query
        if(domain == "movie"):
            prompt_template = NER_MOVIE_PROMPT_TEMPLATE
        elif(domain == "music"):
            prompt_template = NER_MUSIC_PROMPT_TEMPLATE
        elif(domain == "sports"):
            prompt_template = NER_SPORTS_PROMPT_TEMPLATE
        elif(domain == "finance"):
            prompt_template = NER_FINANCE_PROMPT_TEMPLATE
        else:
            prompt_template = NER_OPEN_PROMPT_TEMPLATE 
        final_prompt = prompt_template.format(
            query=query, references=references
        )


        ######################### generation #########################
        # Generate responses via vllm
        result = self.llm.generate(
            final_prompt,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                # Note: We are using 50 max new tokens instead of 75,
                # because the 75 max token limit is checked using the Llama2 tokenizer.
                # The Llama3 model instead uses a differet tokenizer with a larger vocabulary
                # This allows it to represent the same content more efficiently, using fewer tokens.
            ),
            use_tqdm = False
        )
        completion_output = result[0].outputs[0]  # 获取第一个 CompletionOutput
        generated_text = completion_output.text  # 提取生成的文本

        answer = generated_text
        # print("ner answer before process:",trimmed_answer)
        if(domain == "movie"):
            results = self.movie_process_text(answer)
        elif(domain == "music"):
            results = self.music_process_text(answer)
        elif(domain == "sports"):
            results = self.sports_process_text(answer)
        elif(domain == "finance"):
            results = self.finance_process_text(answer)
        else:
            results = self.open_process_text(answer)
        results["time"]=find_date_from_text(query_time, query)
        # print("ner answer:",results)
        return results
    

    
#五个domain的返回值的索引不同，music为person，song，band，sports为soccerteam，nbateam，finance为company，ticker，open为entity
#把该文件放在models目录下，然后
"""
from models.ner import ner
model=ner()
results=model.generate_answer(query,domain)
即可,返回值为字典,key为上面domain对应的索引
"""