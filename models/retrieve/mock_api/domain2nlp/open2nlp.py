from models.mock_api.pycragapi import CRAG

class open2nlp:
    def __init__(self, entity: str = "unknown"):
        self.domain = "open"
        self.entity = entity
        self.api = CRAG()

    def open_entity_name_2_nlp(self, entity_name: str = "unknown"):
        if entity_name == "unknown":
            entity_name_dict = self.api.open_search_entity_by_name(self.entity)
        else:
            entity_name_dict = self.api.open_search_entity_by_name(entity_name)
        if "result" in entity_name_dict and isinstance(
            entity_name_dict["result"], list
        ):
            if entity_name_dict["result"]:
                entity_name = entity_name_dict["result"][0]
        return self._open_entity_2_nlp(entity_name)

    def _open_entity_2_nlp(self, entity_name: str = "unknown"):
        entity_info_dict = self.api.open_get_entity(entity_name)
        entity_info = ""
        if "result" in entity_info_dict and isinstance(
            entity_info_dict["result"], dict
        ):
            entity_info_dict = entity_info_dict["result"]
            entity_info = self.__parse_dict(entity_name, entity_info_dict)

        return entity_info

    def __parse_dict(self, entity_name: str, entity_info_dict: dict):
        entity_info = ""
        for key, value in entity_info_dict.items():
            if isinstance(value, dict):
                if value:
                    entity_info += f"The {key} of {entity_name} is as follows:\n"
                    entity_info += self.__parse_dict(entity_name, value)
                else:
                    entity_info += f"The {key} of {entity_name} is unknown.\n"
            elif isinstance(value, str):
                if value:
                    entity_info += f"The {key} of {entity_name} is {value}.\n"
                else:
                    entity_info += f"The {key} of {entity_name} is unknown.\n"
            elif isinstance(value, list):
                if value:
                    entity_info += f"The {key} of {entity_name} is as follows:\n"
                    entity_info += self.__parse_list(entity_name, value)
                    entity_info += "\n"
                else:
                    entity_info += f"The {key} of {entity_name} is unknown.\n"
        return entity_info

    def __parse_list(self, entity_name: str, entity_info_list: list):
        entity_info = ""
        for value in entity_info_list:
            if isinstance(value, dict):
                if value:
                    entity_info += self.__parse_dict(entity_name, value)
                else:
                    entity_info += f"unknown, "
            elif isinstance(value, str):
                if value:
                    entity_info += f"{value}, "
                else:
                    entity_info += f"unknown, "
            elif isinstance(value, list):
                if value:
                    entity_info += self.__parse_list(entity_name, value)
                else:
                    entity_info += f"unknown, "
        return entity_info


if __name__ == "__main__":
    open_2_nlp = open2nlp("Florida")
    filename = "example_data/open_description.txt"\
    
    description = open_2_nlp.open_entity_name_2_nlp()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(description)