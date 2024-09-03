from models.mock_api.domain2nlp.movie2nlp import movie2nlp
from models.mock_api.domain2nlp.music2nlp import music2nlp
from models.mock_api.domain2nlp.finance2nlp import finance2nlp
from models.mock_api.domain2nlp.sports2nlp import sports2nlp
from models.mock_api.domain2nlp.open2nlp import open2nlp
from models.mock_api.ner import NER
# from models.date import date
from dateutil import parser, tz
import json

class json2nlp:
    def __init__(self, domain_classifier=None, ner_model=None):
        if domain_classifier:
            self.domain_classifier = domain_classifier

        if ner_model:
            self.ner_model = ner_model
        else:
            # 初始化 ner 模型，获取问题中的实体
            self.ner_model = NER()

        # 初始化 date 模型，获取问题所问的真正时间
        # self.date_model = date()

        # 初始化 domain 分类器
        # self.domain_classifier = TODO

        # 初始化各个领域的 json2nlp 模型
        self.movie2nlp = movie2nlp()
        self.music2nlp = music2nlp()
        self.finance2nlp = finance2nlp()
        self.sports2nlp = sports2nlp()
        self.open2nlp = open2nlp()

    def get_knowledge_from_kg(self, query_dict: dict):
        """
        query_dict = {
            'query_time': '2023-07-13 00:00:00 PT', 
            'query': '...'
        }

        return: knowledge = [
        ... ,
        ... ,
        ... 
        ]
        ner_results : dict
        """
        # 获取问题本身
        query = query_dict['query']
        # 获取问题的分类
        domain = self.domain_classifier(query)
        # domain = query_dict['domain']  # 这里之后换成上面的分类器
        # 获取问题中的实体
        ner_results = self.ner_model.generate_answer(query, domain, query_dict['query_time'])
        # 获取问题所问的真正时间
        if 'time' in ner_results and ner_results['time']:
            query_time = ner_results['time']
        else:
            query_time = query_dict['query_time']
        # 选择合适的 json2nlp 模型，生成结果
        knowledge = list()
        if domain == 'movie':
            if 'movie' in ner_results and ner_results['movie']:
                for movie in ner_results['movie']:
                    _ = self.movie2nlp.movie_movie_info_2_nlp(movie)
                    if _:
                        knowledge.append(_)
            if 'person' in ner_results and ner_results['person']:
                for person in ner_results['person']:
                    _ = self.movie2nlp.movie_person_info_2_nlp(person)
                    if _:
                        knowledge.append(_)
            knowledge.append(self.movie2nlp.movie_year_info_2_nlp(self.__extract_year(query_time)))
        elif domain == 'music':
            if 'person' in ner_results and ner_results['person']:
                for person in ner_results['person']:
                    _ = self.music2nlp.music_artist_info_2_nlp(person)
                    if _:
                        knowledge.append(_)
            if 'song' in ner_results and ner_results['song']:
                for song in ner_results['song']:
                    _ = self.music2nlp.music_song_info_2_nlp(song)
                    if _:
                        knowledge.append(_)
            if 'band' in ner_results and ner_results['band']:
                for band in ner_results['band']:
                    _ = self.music2nlp.music_band_info_2_nlp(band)
                    if _:
                        knowledge.append(_)
        elif domain == 'finance':
            if 'company' in ner_results and ner_results['company']:
                for company in ner_results['company']:
                    _ = self.finance2nlp.finance_company_2_nlp(company)
                    if _:
                        knowledge.append(_)
            if 'ticker' in ner_results and ner_results['ticker']:
                for ticker in ner_results['ticker']:
                    _ = self.finance2nlp.finance_ticker_2_nlp(ticker)
                    if _:
                        knowledge.append(_)
        elif domain == 'sports':
            _date = self.__extract_year_month_day(query_time)
            if 'soccerteam' in ner_results and ner_results['soccerteam']:
                for soccerteam in ner_results['soccerteam']:
                    _ = self.sports2nlp.sports_soccer_game_2_nlp(soccerteam, _date)
                    if _:
                        knowledge.append(_)
            if 'nbateam' in ner_results and ner_results['nbateam']:
                for nbateam in ner_results['nbateam']:
                    _ = self.sports2nlp.sports_nba_game_2_nlp(nbateam, _date)
                    if _:
                        knowledge.append(_)
        elif domain == 'open':
            if 'entity' in ner_results and ner_results['entity']:
                for entity in ner_results['entity']:
                    _ = self.open2nlp.open_entity_name_2_nlp(entity)
                    if _:
                        knowledge.append(_)
        
        return knowledge, ner_results

    def __extract_year(self, time: str):
        """
        '2023-07-13 00:00:00 PT' -> 2023
        """
        ft = parser.parse(time)
        pdt = tz.gettz('America/Los_Angeles')
        ft = ft.replace(tzinfo=pdt)
        time = ft.strftime("%Y")
        return time

    def __extract_year_month_day(self, time: str):
        """
        '03/21/2024, 23:51:43 PT' -> '2023-07-13'
        """
        ft = parser.parse(time)
        pdt = tz.gettz('America/Los_Angeles')
        ft = ft.replace(tzinfo=pdt)
        time = ft.strftime("%Y-%m-%d")
        return time


if __name__ == '__main__':
    json_2_nlp = json2nlp()

    query_path = "example_data/crag_task_1_v2_query_only.jsonl"
    output_path = "example_data/query_only_knowledge.jsonl"

    with open(query_path, "r", encoding="utf-8") as f,\
        open(output_path, "w", encoding="utf-8") as f_out:
        for line in f:
            query_dict = json.loads(line)
            knowledge, ner_results = json_2_nlp.get_knowledge_from_kg(query_dict)
            query_dict['knowledge'] = knowledge
            for key, value in ner_results.items():
                query_dict[key] = value
            json.dump(query_dict, f_out)
            f_out.write('\n')


