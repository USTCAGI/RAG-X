from models.mock_api.pycragapi import CRAG

class movie2nlp:
    def __init__(self,  entity: str = "unknown") -> None:
        """
        entity:
        - movie name (str)
        - person name (str)
        - year (int): between 1990 and 2021
        """
        self.domain = "movie"
        self.entity = entity
        self.api = CRAG()

    def movie_person_info_2_nlp(self, person_name: str = "unknown", detailed: bool = False) -> str:
        if person_name == "unknown":
            movie_person_info = self.api.movie_get_person_info(self.entity)
        else:
            movie_person_info = self.api.movie_get_person_info(person_name)
        descriptions = ""
        if "result" in movie_person_info and isinstance(
            movie_person_info["result"], list
        ):
            for person in movie_person_info["result"]:
                descriptions += self._movie_person_dict_2_nlp(person, detailed)
        return descriptions

    def _movie_person_id_2_nlp(self, id: int):
        """
        id: unique id of person

        返回 (name, description)
        这里 description 是一个字符串，包含了 person 的详细信息
        由于只有 movie_movie 会调用这个函数，所以不需要进一步的 detail 信息 (也就是说不需要二级 detail)
        所以这里调用 _movie_person_dict_2_nlp(person, detailed=False) 即可
        """
        movie_person_id_info = self.api.movie_get_person_info_by_id(id)
        name, description = "unknown", ""

        if "result" in movie_person_id_info and isinstance(
            movie_person_id_info["result"], dict
        ):
            person = movie_person_id_info["result"]
            if "name" in person:
                name = person["name"]
            description = self._movie_person_dict_2_nlp(person, detailed=False)
        return name, description

    def _movie_person_dict_2_nlp(self, person: dict, detailed: bool) -> str:
        """
        - name (string): name of person
        - id (int): unique id of person
        - acted_movies (list[int]): list of movie ids in which person acted
        - directed_movies (list[int]): list of movie ids in which person directed
        - birthday (string): string of person's birthday, in the format of "YYYY-MM-DD"
        - oscar_awards: list of oscar awards (dict), win or nominated, in which the movie was the entity. The format for oscar award entity are:
            'year_ceremony' (int): year of the oscar ceremony,
            'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            'category' (string): category of this oscar award,
            'name' (string): name of the nominee,
            'film' (string): name of the film,
            'winner' (bool): whether the person won the award
        """

        MoviePersonPrompt = """{name} is a person who was born on {birthday}. {acted_movies} {directed_movies} {oscar_awards} """
        name, birthday, acted_movies, directed_movies, oscar_awards = (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        )
        if "name" in person:
            name = person["name"]
            name = name if name else "unknown"
        if "birthday" in person:
            birthday = person["birthday"]
            birthday = birthday if birthday else "unknown"
        if "acted_movies" in person:
            acted_movie_ids = person["acted_movies"]
            acted_movies = [self._movie_movie_id_2_nlp(id) for id in acted_movie_ids]
        if "directed_movies" in person:
            directed_movie_ids = person["directed_movies"]
            directed_movies = [
                self._movie_movie_id_2_nlp(id) for id in directed_movie_ids
            ]
        if "oscar_awards" in person:
            oscar_awards = person["oscar_awards"]
            oscar_awards = self._movie_oscar_award_2_nlp(oscar_awards)

        (
            acted_movie_names,
            directed_movie_names,
            acted_movie_descriptions,
            directed_movie_descriptions,
        ) = (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        )
        acted_movie_str, directed_movie_str = "", ""
        if acted_movies != "unknown":
            acted_movie_names = [movie[0] for movie in acted_movies]
            acted_movie_names = ", ".join(acted_movie_names)
            acted_movie_str = f"{name} acted in the movies: {acted_movie_names}. "
        if directed_movies != "unknown":
            directed_movie_names = [movie[0] for movie in directed_movies]
            directed_movie_names = ", ".join(directed_movie_names)
            directed_movie_str = f"{name} directed the movies: {directed_movie_names}. "

        movie_person_descriptions = MoviePersonPrompt.format(
            name=name,
            birthday=birthday,
            acted_movies=acted_movie_str,
            directed_movies=directed_movie_str,
            oscar_awards=oscar_awards,
        )

        if detailed:
            # 如果需要一级详细信息，就在 prompt 后面加上电影的详细信息
            if acted_movies != "unknown":
                acted_movie_descriptions = [movie[1] for movie in acted_movies]
                movie_person_descriptions += " ".join(acted_movie_descriptions)
            if directed_movies != "unknown":
                directed_movie_descriptions = [movie[1] for movie in directed_movies]
                movie_person_descriptions += " ".join(directed_movie_descriptions)
        return movie_person_descriptions

    def movie_movie_info_2_nlp(self, movie_name: str = "unknown", detailed: bool = False) -> str:
        if movie_name == "unknown":
            movie_movie_info = self.api.movie_get_movie_info(self.entity)
        else:
            movie_movie_info = self.api.movie_get_movie_info(movie_name)
        if "result" in movie_movie_info and isinstance(
            movie_movie_info["result"], list
        ):
            descriptions = ""
            for movie in movie_movie_info["result"]:
                descriptions += self._movie_movie_dict_2_nlp(movie, detailed)
        return descriptions

    def _movie_movie_id_2_nlp(self, id: int):
        """
        movie_id: unique id of movie
        """
        movie_movie_id_info = self.api.movie_get_movie_info_by_id(id)
        title, description = "unknown", ""
        if "result" in movie_movie_id_info and isinstance(
            movie_movie_id_info["result"], dict
        ):
            movie = movie_movie_id_info["result"]
            if "title" in movie:
                title = movie["title"]
            description = self._movie_movie_dict_2_nlp(movie, detailed=False)
        return title, description

    def _movie_movie_dict_2_nlp(self, movie: dict, detailed: bool) -> str:
        """
        - title (string): title of movie
        - id (int): unique id of movie
        - release_date (string): string of movie's release date, in the format of "YYYY-MM-DD"
        - length (int): length of movie, in minutes
        - original_title (string): original title of movie, if in another language other than english
        - original_language (string): original language of movie. Example: 'en', 'fr'
        - budget (int): budget of movie, in USD
        - revenue (int): revenue of movie, in USD
        - rating (float): rating of movie, in range [0, 10]
        - genres (list[dict]): list of genres of movie. Sample genre object is  {'id': 123, 'name': 'action'}
        - oscar_awards: list of oscar awards (dict), win or nominated, in which the movie was the entity. The format for oscar award entity are:
            'year_ceremony' (int): year of the oscar ceremony,
            'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            'category' (string): category of this oscar award,
            'name' (string): name of the nominee,
            'film' (string): name of the film,
            'winner' (bool): whether the person won the award
        - cast (list [dict]): list of cast members of the movie and their roles. The format of the cast member entity is:
            'name' (string): name of the cast member,
            'id' (int): unique id of the cast member,
            'character' (string): character played by the cast member in the movie,
            'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
            'order' (int): order of the cast member in the movie. For example, the actress with the lowest order is the main actress,
        - crew' (list [dict]): list of crew members of the movie and their roles. The format of the crew member entity is:
            'name' (string): name of the crew member,
            'id' (int): unique id of the crew member,
            'job' (string): job of the crew member,
        """
        MovieMoviePrompt = """The movie "{title}" was released on {release_date} and duration is {length}mins. The original title of the movie is {original_title}. The original language of the movie is {original_language}. The budget of the movie is ${budget}. The revenue of the movie is ${revenue}. The rating of the movie is {rating}/10. {genres} {oscar_awards} {casts} {crews} """
        (
            title,
            release_date,
            length,
            original_title,
            original_language,
            budget,
            revenue,
            rating,
            genres,
            oscar_awards,
            casts,
            crews,
        ) = (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        )
        if "title" in movie:
            title = movie["title"]
            title = title if title else "unknown"
        if "release_date" in movie:
            release_date = movie["release_date"]
            release_date = release_date if release_date else "unknown"
        if "length" in movie:
            length = movie["length"]
            length = length if length else "unknown"
        if "original_title" in movie:
            original_title = movie["original_title"]
            original_title = original_title if original_title else "unknown"
        if "original_language" in movie:
            original_language = movie["original_language"]
            original_language = original_language if original_language else "unknown"
        if "budget" in movie:
            budget = movie["budget"]
            budget = budget if budget else "unknown"
        if "revenue" in movie:
            revenue = movie["revenue"]
            revenue = revenue if revenue else "unknown"
        if "rating" in movie:
            rating = movie["rating"]
            rating = rating if rating else "unknown"
        if "genres" in movie:
            genres = movie["genres"]
            genres = self._movie_genres_2_nlp(title, genres)
        if "oscar_awards" in movie:
            oscar_awards = movie["oscar_awards"]
            oscar_awards = self._movie_oscar_award_2_nlp(oscar_awards)
        if "cast" in movie:
            casts = movie["cast"]
            casts = self._movie_cast_2_nlp(title, casts, detailed)
        if "crew" in movie:
            crews = movie["crew"]
            crews = self._movie_crew_2_nlp(title, crews)

        movie_descriptions = MovieMoviePrompt.format(
            title=title,
            release_date=release_date,
            length=length,
            original_title=original_title,
            original_language=original_language,
            budget=budget,
            revenue=revenue,
            rating=rating,
            genres=genres,
            oscar_awards=oscar_awards,
            casts=casts,
            crews=crews,
        )
        return movie_descriptions

    def _movie_genres_2_nlp(self, title: str, genres: list) -> str:
        """
        - genres (list[dict]): list of genres of movie. Sample genre object is  {'id': 123, 'name': 'action'}
        """
        MovieGenresPrompt = """The movie "{title}" belongs to the genre {genre}. """
        all_genres = list()
        for genre in genres:
            if "name" in genre:
                genre_name = genre["name"]
                genre_name = genre_name if genre_name else "unknown"
                all_genres.append(genre_name)

        all_genres = ", ".join(all_genres)
        return MovieGenresPrompt.format(title=title, genre=all_genres)

    def _movie_oscar_award_2_nlp(self, oscar_awards: list) -> str:
        """
        - oscar_awards: list of oscar awards (dict), win or nominated, in which the movie was the entity. The format for oscar award entity are:
            'year_ceremony' (int): year of the oscar ceremony,
            'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            'category' (string): category of this oscar award,
            'name' (string): name of the nominee,
            'film' (string): name of the film,
            'winner' (bool): whether the person won the award
        """
        WinOscarAwardPrompt = """At the {ceremony}th oscar ceremony in {year_ceremony}, {name} won the {category} award for the film "{film}". """
        NominateOscarAwardPrompt = """At the {ceremony}th oscar ceremony in {year_ceremony}, {name} was nominated for the {category} award for the film "{film}". """
        descriptions = ""
        for oscar_award in oscar_awards:
            is_winner = False
            if "winner" in oscar_award:
                is_winner = oscar_award["winner"]
            year_ceremony, ceremony, category, name, film = (
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
            )
            if "year_ceremony" in oscar_award:
                year_ceremony = oscar_award["year_ceremony"]
                year_ceremony = year_ceremony if year_ceremony else "unknown"
            if "ceremony" in oscar_award:
                ceremony = oscar_award["ceremony"]
                ceremony = ceremony if ceremony else "unknown"
            if "category" in oscar_award:
                category = oscar_award["category"]
                category = category if category else "unknown"
            if "name" in oscar_award:
                name = oscar_award["name"]
                name = name if name else "unknown"
            if "film" in oscar_award:
                film = oscar_award["film"]
                film = film if film else "unknown"

            if is_winner:
                descriptions += WinOscarAwardPrompt.format(
                    ceremony=ceremony,
                    year_ceremony=year_ceremony,
                    category=category,
                    name=name,
                    film=film,
                )
            else:
                descriptions += NominateOscarAwardPrompt.format(
                    ceremony=ceremony,
                    year_ceremony=year_ceremony,
                    category=category,
                    name=name,
                    film=film,
                )
        return descriptions

    def _movie_cast_2_nlp(self, title: str, casts: list, detailed) -> str:
        """
        - cast (list [dict]): list of cast members of the movie and their roles. The format of the cast member entity is:
            'name' (string): name of the cast member,
            'id' (int): unique id of the cast member,
            'character' (string): character played by the cast member in the movie,
            'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
            'order' (int): order of the cast member in the movie. For example, the actress with the lowest order is the main actress
        """
        MovieCastPrompt = """The number {order} character {character} in the movie "{title}" is played by {name}, whose gender is {gender}. """
        descriptions = ""
        for cast in casts:
            name, id, character, gender, order = (
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
            )
            if "name" in cast:
                name = cast["name"]
                name = name if name else "unknown"
            if "id" in cast:
                id = cast["id"]
                id = id if id else "unknown"
            if "character" in cast:
                character = cast["character"]
                character = character if character else "unknown"
            if "gender" in cast:
                gender = cast["gender"]
                # 2 for actor, 1 for actress, 0 for unknown
                if gender == 0:
                    gender = "unknown"
                else:
                    gender = "male" if gender == 2 else "female"
            if "order" in cast:
                order = cast["order"]
                if isinstance(order, int):
                    order = order + 1  # 从 0 开始，所以加 1
                else:
                    order = "unknown"
            movie_cast_description = MovieCastPrompt.format(
                order=order,
                character=character,
                title=title,
                name=name,
                gender=gender,
            )

            if detailed:
                # 如果需要一级详细信息，就在 prompt 后面加上演员的详细信息
                _, character_description = self._movie_person_id_2_nlp(id)
                movie_cast_description += character_description

            descriptions += movie_cast_description
        return descriptions

    def _movie_crew_2_nlp(self, title, crews):
        """
        - crew' (list [dict]): list of crew members of the movie and their roles. The format of the crew member entity is:
            'name' (string): name of the crew member,
            'id' (int): unique id of the crew member,
            'job' (string): job of the crew member,
        """
        MovieCrewPrompt = """{name} is the {job} of the movie "{title}". """
        descriptions = ""
        for crew in crews:
            name, job = "unknown", "unknown"
            if "name" in crew:
                name = crew["name"]
                name = name if name else "unknown"
            if "job" in crew:
                job = crew["job"]
                job = job if job else "unknown"
            descriptions += MovieCrewPrompt.format(name=name, title=title, job=job)
        return descriptions

    def movie_year_info_2_nlp(self, year: int, detailed: bool = False) -> str:
        """
        - movie_list: list of movie ids in the year. This field can be very long to a few thousand films
        - oscar_awards: list of oscar awards (dict), held in that particular year. The format for oscar award entity are:
            'year_ceremony' (int): year of the oscar ceremony,
            'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            'category' (string): category of this oscar award,
            'name' (string): name of the nominee,
            'film' (string): name of the film,
            'winner' (bool): whether the person won the award
        """
        # Note that we only support years between 1990 and 2021
        if year == "unknown":
            year = self.entity
        
        if not isinstance(year, int) or year < 1990 or year > 2021:
            return ""

        MovieYearInfoPrompt = """In the year {year}, the following movies were released: {movies}. {oscar_awards} """
        movies, oscar_awards = ("unknown", "unknown")

        movie_year_info = self.api.movie_get_year_info(year)
        if "result" in movie_year_info and isinstance(movie_year_info["result"], dict):
            year_info = movie_year_info["result"]
            if "movie_list" in year_info:
                movie_list = year_info["movie_list"]
                movies = [self._movie_movie_id_2_nlp(id) for id in movie_list]
            if "oscar_awards" in year_info:
                oscar_awards = year_info["oscar_awards"]
                oscar_awards = self._movie_oscar_award_2_nlp(oscar_awards)

        movie_names, movie_descriptions = ("unknown", "unknown")
        if movies != "unknown":
            movie_names = [movie[0] for movie in movies]
            movie_names = ", ".join(movie_names)
        movie_year_info_descriptions = MovieYearInfoPrompt.format(
            year=year, movies=movie_names, oscar_awards=oscar_awards
        )

        if detailed:
            # 如果需要一级详细信息，就在 prompt 后面加上电影的详细信息
            movie_descriptions = [movie[1] for movie in movies]
            movie_year_info_descriptions += " ".join(movie_descriptions)
        return movie_year_info_descriptions


if __name__ == "__main__":
    movie_2_nlp = movie2nlp("Lord of the Rings")
    is_detailed = False
    filename = "example_data/movie_description"
    if is_detailed:
        filename += "_detailed.txt"
    else:
        filename += "_no_detailed.txt"

    description = (
        movie_2_nlp.movie_year_info_2_nlp(detailed=is_detailed)
        + movie_2_nlp.movie_movie_info_2_nlp(detailed=is_detailed)
        + movie_2_nlp.movie_person_info_2_nlp(detailed=is_detailed)
    )
    # 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(description)
