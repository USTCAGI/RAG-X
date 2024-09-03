from models.mock_api.pycragapi import CRAG
import re
from collections import defaultdict
from ast import literal_eval

class sports2nlp:
    def __init__(self, entity: str = "unknown", date: str = ""):
        """
        date: in format of %Y-%m-%d, %Y-%m or %Y, e.g. 2024-03-01, 2024-03, 2024
        """
        self.domain = "sports"
        self.entity = entity
        self.date = date
        self.api = CRAG()

    def __check_date(self, date):
        """
        check if date in format of %Y-%m-%d, %Y-%m or %Y, e.g. 2024-03-01, 2024-03, 2024
        return True if date is in format, otherwise return False
        """
        return bool(re.match(r"^\d{4}(-\d{2}){0,2}$", date))


    def sports_soccer_game_2_nlp(self, team_name: str = "unknown", date: str = "unknown"):
        team_name = self.entity if team_name == "unknown" else team_name
        date = self.date if date == "unknown" else date

        is_date_valid = bool(date and self.__check_date(date))
        if is_date_valid:
            team_name_dict = self.api.sports_soccer_get_games_on_date(date, team_name)
        else:
            team_name_dict = self.api.sports_soccer_get_games_on_date(team_name)
        
        if "result" in team_name_dict and isinstance(
            team_name_dict["result"], dict
        ):
            if team_name_dict["result"]:
                team_name_dict = team_name_dict["result"]

        
        team_name_dict = self._transform_by_team(team_name_dict)
        teams_info = ""
        for team, info in team_name_dict.items():
            teams_info += f"At date {date}, {team} has a match information as follows:\n"
            for key, value in info.items():
                teams_info += f"The {key} is {value}\n"
        
        return teams_info

    def sports_nba_game_2_nlp(self, team_name: str = "unknown", date: str = "unknown"):
        team_name = self.entity if team_name == "unknown" else team_name
        date = self.date if date == "unknown" else date

        is_date_valid = bool(date and self.__check_date(date))
        if is_date_valid:
            team_name_dict = self.api.sports_nba_get_games_on_date(date, team_name)
        else:
            team_name_dict = self.api.sports_nba_get_games_on_date(team_name)
        
        if "result" in team_name_dict and isinstance(
            team_name_dict["result"], dict
        ):
            if team_name_dict["result"]:
                team_name_dict = team_name_dict["result"]

        
        team_name_dict = self._transform_by_team(team_name_dict)
        teams_info = ""
        for team, info in team_name_dict.items():
            teams_info += f"At date {date}, {team} has a match information as follows:\n"
            for key, value in info.items():
                teams_info += f"The {key} is {value}\n"
        
        return teams_info

    def __extract_soccer_match_info(self, match_info_str: str):
        """
        "('ENG-Premier League', '2324', 'Everton', '2024-03-09 Manchester Utd-Everton')"
        ->
        (league_name, team_name, home_team)
        """
        match_tuple = literal_eval(match_info_str)
        league_name = match_tuple[0]
        team_name = match_tuple[2]
        home_team = match_tuple[3].split(" ")[1]
        return league_name, team_name, home_team

    def _transform_by_team(self, team_name_dict: dict):
        """
        'season_id': {'0': '12022',
        '1': '12022',
        '2': '12022',
        '3': '12022',
        '4': '12022'}
        ->
        {
        '0': {'season_id': '12022', },
        '1': {'season_id': '12022', },
        '2': {'season_id': '12022', },
        '3': {'season_id': '12022', }
        }
        """
        is_nba = False
        nba_id_2_name = dict()
        if "team_name_home" in team_name_dict:
            is_nba = True
            if team_name_dict["team_name_home"]:
                nba_id_2_name = team_name_dict["team_name_home"]
        
        transformed_team_name_dict = defaultdict(dict)
        for key, value in team_name_dict.items():
            if isinstance(value, dict):
                for team, true_info in value.items():
                    if is_nba:
                        # 如果是 nba，则多一步转化
                        if team in nba_id_2_name:
                            team = nba_id_2_name[team]
                    else:
                        league_name, team_name, home_team = self.__extract_soccer_match_info(team)
                        team = team_name
                        if league_name:
                            transformed_team_name_dict[team]["league_name"] = league_name
                        if team_name:
                            transformed_team_name_dict[team]["team_name"] = team_name
                        if home_team:
                            transformed_team_name_dict[team]["home_team"] = home_team
                    if true_info:
                        transformed_team_name_dict[team][key] = true_info
        
        return transformed_team_name_dict


if __name__ == "__main__":
    sports_2_nlp = sports2nlp("Everton", "2024-03-09")
    filename = "example_data/sports_description.txt"

    description = sports_2_nlp.sports_soccer_game_2_nlp()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(description)