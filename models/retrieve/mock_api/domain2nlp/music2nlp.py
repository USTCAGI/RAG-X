from models.mock_api.pycragapi import CRAG
from collections import defaultdict
import pycountry


# 字典树的简单实现，用于歌曲/专辑名与艺术家所有作品的前缀匹配
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.end_of_word = True

    def starts_with(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class music2nlp:
    def __init__(self, entity: str = "unknown") -> None:
        """
        entity:
        - artist name
        - song name
        - band name
        - rank
        - date
        - year
        """
        self.domain = "music"
        self.entity = entity
        self.api = CRAG()
        # 1958-2019年格莱美奖的所有最佳新人奖、年度歌曲奖、年度专辑奖
        self.grammy_best_new_artist_all_years = defaultdict(list)
        self.grammy_best_song_all_years = defaultdict(list)
        self.grammy_best_album_all_years = defaultdict(list)
        self._set_grammy_best_info_all_years()

    def _set_grammy_best_info_all_years(self):
        """
        for all years between 1958 to 2019
        {"artist name": [year1, year2, ...]}
        {"song name": [year1, year2, ...]}
        {"album name": [year1, year2, ...]}
        """
        for year in range(1958, 2020):
            grammy_best_new_artist_year = self.api.music_grammy_get_best_artist_by_year(
                year
            )
            if "result" in grammy_best_new_artist_year and isinstance(
                grammy_best_new_artist_year["result"], list
            ):
                for artist in grammy_best_new_artist_year["result"]:
                    self.grammy_best_new_artist_all_years[artist].append(year)
            grammy_best_song_year = self.api.music_grammy_get_best_song_by_year(year)
            if "result" in grammy_best_song_year and isinstance(
                grammy_best_song_year["result"], list
            ):
                for song in grammy_best_song_year["result"]:
                    self.grammy_best_song_all_years[song].append(year)
            grammy_best_album_year = self.api.music_grammy_get_best_album_by_year(year)
            if "result" in grammy_best_album_year and isinstance(
                grammy_best_album_year["result"], list
            ):
                for album in grammy_best_album_year["result"]:
                    self.grammy_best_album_all_years[album].append(year)

    def __get_country_name(self, country_code):
        try:
            country_name = pycountry.countries.get(alpha_2=country_code).name
        except:
            country_name = "unknown"
        return country_name

    def music_artist_info_2_nlp(self, artist_name: str = "unknown"):
        if artist_name == "unknown":
            # 如果没有提供艺术家名，则根据类中的实体名搜索艺术家名
            artist_name_dict = self.api.music_search_artist_entity_by_name(self.entity)
        else:
            # 如果提供了艺术家名，则直接搜索
            artist_name_dict = self.api.music_search_artist_entity_by_name(artist_name)

        if "result" in artist_name_dict and isinstance(
            artist_name_dict["result"], list
        ):
            if artist_name_dict["result"]:
                artist_name = artist_name_dict["result"][0]

        birth_place_dict = self.api.music_get_artist_birth_place(artist_name)
        birth_place = "unknown"
        if "result" in birth_place_dict and isinstance(birth_place_dict["result"], str):
            birth_place = birth_place_dict["result"]
        birth_place = self.__get_country_name(birth_place)  # 将国家代码转换为国家名
        birth_date_dict = self.api.music_get_artist_birth_date(artist_name)
        birth_date = "unknown"
        if "result" in birth_date_dict and isinstance(birth_date_dict["result"], str):
            birth_date = birth_date_dict["result"]
        lifespan_dict = self.api.music_get_lifespan(artist_name)
        lifespan_born, lifespan_die = "unknown", "unknown"
        if "result" in lifespan_dict and isinstance(lifespan_dict["result"], list):
            if len(lifespan_dict["result"]) == 2:
                lifespan_born, lifespan_die = lifespan_dict["result"]
                if lifespan_born is None:
                    lifespan_born = "unknown"
                if lifespan_die is None:
                    lifespan_die = "unknown"

        all_works_dict = self.api.music_get_artist_all_works(artist_name)
        all_works = list()
        if "result" in all_works_dict and isinstance(all_works_dict["result"], list):
            all_works = all_works_dict["result"]
        all_works_str = ", ".join(all_works)
        # 用字典树存储所有作品，用于后续匹配
        all_works_trie = Trie()
        for work in all_works:
            all_works_trie.insert(work)

        ArtistBaseInfoPrompt = """{artist_name} was born in {birth_place} on {birth_date}. {artist_name} was born on {lifespan_born} and died on {lifespan_die}. {artist_name} has produced {all_works_str}.
"""
        artist_base_info = ArtistBaseInfoPrompt.format(
            artist_name=artist_name,
            birth_place=birth_place,
            birth_date=birth_date,
            lifespan_born=lifespan_born,
            lifespan_die=lifespan_die,
            all_works_str=all_works_str,
        )

        # Grammy
        # 该艺术家获得格莱美奖的次数（1958-2019）
        grammy_artist_count = 0
        grammy_artist_count_dict = self.api.music_grammy_get_award_count_by_artist(
            artist_name
        )
        if "result" in grammy_artist_count_dict and isinstance(
            grammy_artist_count_dict["result"], int
        ):
            grammy_artist_count = grammy_artist_count_dict["result"]
        if grammy_artist_count:
            grammy_artist_count_str = f"{artist_name} has won {grammy_artist_count} Grammy Awards during 1958-2019."
        else:
            # grammy_artist_count_str = f"{artist_name} has not won any Grammy Awards during 1958-2019."
            grammy_artist_count_str = ""

        # 该艺术家获得格莱美奖的年份（1958-2019），可能不全
        grammy_artist_years_list = list()
        grammy_artist_years_dict = self.api.music_grammy_get_award_date_by_artist(
            artist_name
        )
        if "result" in grammy_artist_years_dict and isinstance(
            grammy_artist_years_dict["result"], list
        ):
            grammy_artist_years_list = grammy_artist_years_dict["result"]
        if grammy_artist_years_list:
            grammy_artist_years_str = f"{artist_name} has won Grammy Awards in {', '.join(str(year) for year in grammy_artist_years_list)} during 1958-2019."
        else:
            # grammy_artist_years_str = f"{artist_name} has not won any Grammy Awards during 1958-2019."
            grammy_artist_years_str = ""

        # 该艺术家获得格莱美奖的最佳新人奖的年份（1958-2019）
        grammy_best_new_artist_years_list = list()
        if artist_name in self.grammy_best_new_artist_all_years:
            grammy_best_new_artist_years_list = self.grammy_best_new_artist_all_years[
                artist_name
            ]
        if grammy_best_new_artist_years_list:
            grammy_best_new_artist_years_str = f"{artist_name} has won the Best New Artist Award at the Grammy Awards in {', '.join(str(year) for year in grammy_best_new_artist_years_list)} during 1958-2019."
        else:
            grammy_best_new_artist_years_str = f"{artist_name} has not won the Best New Artist Award at the Grammy Awards during 1958-2019."

        # 该艺术家获得格莱美年度歌曲奖的歌曲和年份（1958-2019）
        grammy_best_song_by_artist_years_dict = defaultdict(list)
        for best_song, years in self.grammy_best_song_all_years.items():
            # 如果 best_song 是该艺术家的所有作品中的某个作品的前缀，则认为该歌曲是该艺术家的作品
            if all_works_trie.starts_with(best_song):
                grammy_best_song_by_artist_years_dict[best_song] = years
        if grammy_best_song_by_artist_years_dict:
            grammy_best_song_by_artist_years_str = (
                f'{artist_name} has won the Grammy Award for "Song of the Year" '
            )
            for best_song, years in grammy_best_song_by_artist_years_dict.items():
                grammy_best_song_by_artist_years_str += (
                    f"for \"{best_song}\" in {', '.join(str(year) for year in years)} "
                )
            grammy_best_song_by_artist_years_str += "during 1958-2019."
        else:
            # grammy_best_song_by_artist_years_str = f'{artist_name} has not won the Grammy Award for "Song of the Year" during 1958-2019.'
            grammy_best_song_by_artist_years_str = ""

        # 该艺术家获得格莱美年度专辑奖的专辑和年份（1958-2019）
        grammy_best_album_by_artist_years_dict = defaultdict(list)
        for best_album, years in self.grammy_best_album_all_years.items():
            # 如果 best_album 是该艺术家的所有作品中的某个作品的前缀，则认为该专辑是该艺术家的作品
            if all_works_trie.starts_with(best_album):
                grammy_best_album_by_artist_years_dict[best_album] = years
        if grammy_best_album_by_artist_years_dict:
            grammy_best_album_by_artist_years_str = (
                f'{artist_name} has won the Grammy Award for "Album of the Year" '
            )
            for best_album, years in grammy_best_album_by_artist_years_dict.items():
                grammy_best_album_by_artist_years_str += (
                    f"for \"{best_album}\" in {', '.join(str(year) for year in years)} "
                )
            grammy_best_album_by_artist_years_str += "during 1958-2019."
        else:
            # grammy_best_album_by_artist_years_str = f'{artist_name} has not won the Grammy Award for "Album of the Year" during 1958-2019.'
            grammy_best_album_by_artist_years_str = ""

        ArtistGrammyInfoPrompt = f"""{grammy_artist_count_str} {grammy_artist_years_str} {grammy_best_new_artist_years_str} {grammy_best_song_by_artist_years_str} {grammy_best_album_by_artist_years_str}"""
        artist_grammy_info = ArtistGrammyInfoPrompt.format(
            grammy_artist_count_str=grammy_artist_count_str,
            grammy_artist_years_str=grammy_artist_years_str,
            grammy_best_new_artist_years_str=grammy_best_new_artist_years_str,
            grammy_best_song_by_artist_years_str=grammy_best_song_by_artist_years_str,
            grammy_best_album_by_artist_years_str=grammy_best_album_by_artist_years_str,
        )

        return artist_base_info  + "\n"+ artist_grammy_info

    def music_song_info_2_nlp(self, song_name: str = "unknown"):
        if song_name == "unknown":
            # 如果没有提供歌曲名，则根据实体名搜索歌曲名
            song_name_dict = self.api.music_search_song_entity_by_name(self.entity)
        else:
            # 如果提供了歌曲名，则直接搜索
            song_name_dict = self.api.music_search_song_entity_by_name(song_name)
        if "result" in song_name_dict and isinstance(song_name_dict["result"], list):
            if song_name_dict["result"]:
                song_name = song_name_dict["result"][0]

        song_author_dict = self.api.music_get_song_author(song_name)
        song_author = "unknown"
        if "result" in song_author_dict and isinstance(song_author_dict["result"], str):
            song_author = song_author_dict["result"]

        song_release_date_dict = self.api.music_get_song_release_date(song_name)
        song_release_date = "unknown"
        if "result" in song_release_date_dict and isinstance(
            song_release_date_dict["result"], str
        ):
            song_release_date = song_release_date_dict["result"]

        song_release_country_dict = self.api.music_get_song_release_country(song_name)
        song_release_country = "unknown"
        if "result" in song_release_country_dict and isinstance(
            song_release_country_dict["result"], str
        ):
            song_release_country = song_release_country_dict["result"]
        song_release_country = self.__get_country_name(song_release_country)

        SongBaseInfoPrompt = f"""{song_name} was written by {song_author} and released on {song_release_date} in {song_release_country}."""
        song_base_info = SongBaseInfoPrompt.format(
            song_name=song_name,
            song_author=song_author,
            song_release_date=song_release_date,
            song_release_country=song_release_country,
        )

        # Grammy
        # 该歌曲获得格莱美奖的次数（1958-2019）
        grammy_song_count = 0
        grammy_song_count_dict = self.api.music_grammy_get_award_count_by_song(
            song_name
        )
        if "result" in grammy_song_count_dict and isinstance(
            grammy_song_count_dict["result"], int
        ):
            grammy_song_count = grammy_song_count_dict["result"]
        if grammy_song_count:
            grammy_song_count_str = f"{song_name} has won {grammy_song_count} Grammy Awards during 1958-2019."
        else:
            grammy_song_count_str = (
                f"{song_name} has not won any Grammy Awards during 1958-2019."
            )

        # 该歌曲获得年度歌曲奖的年份（1958-2019）
        grammy_best_song_years_list = list()
        if song_name in self.grammy_best_song_all_years:
            grammy_best_song_years_list = self.grammy_best_song_all_years[song_name]
        if grammy_best_song_years_list:
            grammy_best_song_years_str = f"{song_name} has won the Grammy Award for \"Song of the Year\" in {', '.join(str(year) for year in grammy_best_song_years_list)} during 1958-2019."
        else:
            grammy_best_song_years_str = f'{song_name} has not won the Grammy Award for "Song of the Year" during 1958-2019.'
        
        # 该歌曲获得年度唱片奖的年份（1958-2019）
        grammy_best_album_years_list = list()
        if song_name in self.grammy_best_album_all_years:
            grammy_best_album_years_list = self.grammy_best_album_all_years[song_name]
        if grammy_best_album_years_list:
            grammy_best_album_years_str = f"{song_name} has won the Grammy Award for \"Album of the Year\" in {', '.join(str(year) for year in grammy_best_album_years_list)} during 1958-2019."
        else:
            grammy_best_album_years_str = f'{song_name} has not won the Grammy Award for "Album of the Year" during 1958-2019.'

        SongGrammyInfoPrompt = (
            f"""{grammy_song_count_str} {grammy_best_song_years_str} {grammy_best_album_years_str}"""
        )
        song_grammy_info = SongGrammyInfoPrompt.format(
            grammy_song_count_str=grammy_song_count_str,
            grammy_best_song_years_str=grammy_best_song_years_str,
            grammy_best_album_years_str=grammy_best_album_years_str,
        )

        return song_base_info + "\n" + song_grammy_info

        # Billboard TODO

    def music_band_info_2_nlp(self, band_name: str = "unknown", detailed: bool = True):
        if band_name == "unknown":
            # 如果没有提供乐队名，则根据实体名搜索乐队名
            band_name_dict = self.api.music_search_artist_entity_by_name(self.entity)
        else:
            # 如果提供了乐队名，则直接搜索
            band_name_dict = self.api.music_search_artist_entity_by_name(band_name)
        if "result" in band_name_dict and isinstance(band_name_dict["result"], list):
            if band_name_dict["result"]:
                band_name = band_name_dict["result"][0]

        band_members_dict = self.api.music_get_members(band_name)
        band_members = list()
        if "result" in band_members_dict and isinstance(
            band_members_dict["result"], list
        ):
            band_members = band_members_dict["result"]
        band_members_str = ", ".join(band_members)

        BandBaseInfoPrompt = f"""{band_name} consists of {band_members_str}."""
        band_base_info = BandBaseInfoPrompt.format(
            band_name=band_name, band_members_str=band_members_str
        )

        # 如果需要详细信息
        if detailed:
            for member in band_members:
                member_base_info = self.music_artist_info_2_nlp(member)
                band_base_info += member_base_info

        return band_base_info


if __name__ == "__main__":
    music_2_nlp = music2nlp("Taylor Swift")
    filename = "example_data/music_description.txt"

    description = music_2_nlp.music_artist_info_2_nlp()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(description)
