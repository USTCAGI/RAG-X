from models.mock_api.pycragapi import CRAG
from dateutil import parser, tz


class finance2nlp:
    def __init__(self, entity: str = "unknown"):
        self.domain = "finance"
        self.entity = entity
        self.api = CRAG()

    def finance_company_2_nlp(self, company_name: str = "unknown"):
        if company_name == "unknown":
            company_name_dict = self.api.finance_get_company_name(self.entity)
        else:
            company_name_dict = self.api.finance_get_company_name(company_name)
        if "result" in company_name_dict and isinstance(
            company_name_dict["result"], list
        ):
            if company_name_dict["result"]:
                company_name = company_name_dict["result"][0]

        ticker_dict = self.api.finance_get_ticker_by_name(company_name)
        ticker_name = "unknown"
        if "result" in ticker_dict and isinstance(ticker_dict["result"], str):
            ticker_name = ticker_dict["result"]

        CompanyBaseInfoPrompt = (
            """{company_name} is a company with ticker name {ticker_name}"""
        )
        company_base_info = CompanyBaseInfoPrompt.format(
            company_name=company_name, ticker_name=ticker_name
        )

        ticker_info = self.finance_ticker_2_nlp(ticker_name)
        ticker_price_history = self.finance_ticker_price_history_2_nlp(ticker_name)

        return f"{company_base_info}\n{ticker_info}\n{ticker_price_history}"

    def __convert_time(self, time: str):
        """
        '2023-07-13, 00:00:00 EST' -> '2023-07-12, 21:00:00 PDT Wednesday'
        """
        try:
            ft = parser.parse(time)
            est = tz.gettz("America/New_York")
            pdt = tz.gettz("America/Los_Angeles")
            ft = ft.replace(tzinfo=est)
            ft = ft.astimezone(pdt)
            time = ft.strftime("%Y-%m-%d, %H:%M:%S %Z %A")
        except:
            pass
        return time

    def finance_ticker_2_nlp(
        self, ticker_name: str = "unknown", detailed: bool = False
    ):
        dividends_dict = self.api.finance_get_dividends_history(ticker_name)
        dividends_str = "unknown"
        if "result" in dividends_dict and isinstance(dividends_dict["result"], dict):
            dividends_dict = dividends_dict["result"]
            """
            example:
            {'2019-12-19 00:00:00 EST': 0.058,
            '2020-03-19 00:00:00 EST': 0.2,
            '2020-06-12 00:00:00 EST': 0.2,
            ...
            }
            """
            if dividends_dict:
                dividends_str = (
                    f"The dividend information of {ticker_name} is as follows:\n"
                )
                for time, dividend in dividends_dict.items():
                    time = self.__convert_time(time)
                    dividends_str += f"At {time}, the dividend is {dividend}\n"
            else:
                dividends_str = f"The dividend information of {ticker_name} is unknown"
        else:
            dividends_str = f"The dividend information of {ticker_name} is unknown"

        market_cap_dict = self.api.finance_get_market_capitalization(ticker_name)
        market_cap = "unknown"
        if "result" in market_cap_dict and isinstance(market_cap_dict["result"], float):
            market_cap = market_cap_dict["result"]
        market_cap_str = f"The market capitalization of {ticker_name} is {market_cap}"

        eps_dict = self.api.finance_get_eps(ticker_name)
        eps = "unknown"
        if "result" in eps_dict and isinstance(eps_dict["result"], float):
            eps = eps_dict["result"]
        eps_str = f"The earning per share of {ticker_name} is {eps}"

        pe_ratio_dict = self.api.finance_get_pe_ratio(ticker_name)
        pe_ratio = "unknown"
        if "result" in pe_ratio_dict and isinstance(pe_ratio_dict["result"], float):
            pe_ratio = pe_ratio_dict["result"]
        pe_ratio_str = f"The price-earning ratio of {ticker_name} is {pe_ratio}"

        ticker_info_dict = self.api.finance_get_info(ticker_name)
        ticker_info_str = "unknown"
        if "result" in ticker_info_dict and isinstance(
            ticker_info_dict["result"], dict
        ):
            ticker_info_dict = ticker_info_dict["result"]

            if ticker_info_dict:
                ticker_info_str = ""
                for key, value in ticker_info_dict.items():
                    if key == "companyOfficers":
                        ticker_info_str += self._get_company_officers(ticker_name, value)
                    else:
                        ticker_info_str += f"The {key} of {ticker_name} is {value}\n"
            else:
                ticker_info_str = f"The other information of {ticker_name} is unknown"
        else:
            ticker_info_str = f"The other information of {ticker_name} is unknown"

        return f"{dividends_str}\n{market_cap_str}\n{eps_str}\n{pe_ratio_str}\n{ticker_info_str}"

    def _get_company_officers(self, ticker_name, company_officers):
        ticker_company_officers_str = (
            f"The company officers of {ticker_name} are as follows:\n"
        )
        CompanyOfficerPrompt = """The officer {name} is {age} years old, with title {title}. {name} was born in {yearBorn}. In fiscal year {fiscalYear}, {name} received total pay of {totalPay}, exercised value of {exercisedValue}, and unexercised value of {unexercisedValue}."""
        for officer in company_officers:
            (
                maxAge,
                name,
                age,
                title,
                yearBorn,
                fiscalYear,
                totalPay,
                exercisedValue,
                unexercisedValue,
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
            )
            if "maxAge" in officer:
                maxAge = officer["maxAge"]
                maxAge = maxAge if maxAge else "unknown"
            if "name" in officer:
                name = officer["name"]
                name = name if name else "unknown"
            if "age" in officer:
                age = officer["age"]
                age = age if age else "unknown"
            if "title" in officer:
                title = officer["title"]
                title = title if title else "unknown"
            if "yearBorn" in officer:
                yearBorn = officer["yearBorn"]
                yearBorn = yearBorn if yearBorn else "unknown"
            if "fiscalYear" in officer:
                fiscalYear = officer["fiscalYear"]
                fiscalYear = fiscalYear if fiscalYear else "unknown"
            if "totalPay" in officer:
                totalPay = officer["totalPay"]
                totalPay = totalPay if totalPay else "unknown"
            if "exercisedValue" in officer:
                exercisedValue = officer["exercisedValue"]
                # 这里如果是 0 也是有意义的，所以不能直接 if exercisedValue
                if exercisedValue is not None:
                    exercisedValue = exercisedValue
                else:
                    exercisedValue = "unknown"
            if "unexercisedValue" in officer:
                unexercisedValue = officer["unexercisedValue"]
                if unexercisedValue is not None:
                    unexercisedValue = unexercisedValue
                else:
                    unexercisedValue = "unknown"
            ticker_company_officers_str += CompanyOfficerPrompt.format(
                name=name,
                age=age,
                title=title,
                yearBorn=yearBorn,
                fiscalYear=fiscalYear,
                totalPay=totalPay,
                exercisedValue=exercisedValue,
                unexercisedValue=unexercisedValue,
            )

        return ticker_company_officers_str

    def finance_ticker_price_history_2_nlp(
        self, ticker_name: str = "unknown", detailed: bool = False
    ):
        if detailed:
            ticker_price_history_dict = self.api.finance_get_detailed_price_history(
                ticker_name
            )
        else:
            ticker_price_history_dict = self.api.finance_get_price_history(ticker_name)

        ticker_price_history_str = "unknown"
        if "result" in ticker_price_history_dict and isinstance(
            ticker_price_history_dict["result"], dict
        ):
            ticker_price_history_dict = ticker_price_history_dict["result"]

            if ticker_price_history_dict:
                ticker_price_history_str = (
                    f"The price history of {ticker_name} is as follows:\n"
                )
                TickerPricePrompt = """At {time}, the open price is {open_price}, the close price is {close_price}, the high price is {high_price}, the low price is {low_price}, and the volume is {volume}."""
                for time, price in ticker_price_history_dict.items():
                    time = self.__convert_time(time)
                    open_price, close_price, high_price, low_price, volume = (
                        "unknown",
                        "unknown",
                        "unknown",
                        "unknown",
                        "unknown",
                    )
                    if isinstance(price, dict):
                        if "Open" in price:
                            open_price = price["Open"]
                            open_price = open_price if open_price else "unknown"
                        if "Close" in price:
                            close_price = price["Close"]
                            close_price = close_price if close_price else "unknown"
                        if "High" in price:
                            high_price = price["High"]
                            high_price = high_price if high_price else "unknown"
                        if "Low" in price:
                            low_price = price["Low"]
                            low_price = low_price if low_price else "unknown"
                        if "Volume" in price:
                            volume = price["Volume"]
                            volume = volume if volume else "unknown"

                    ticker_price_history_str += TickerPricePrompt.format(
                        time=time,
                        open_price=open_price,
                        close_price=close_price,
                        high_price=high_price,
                        low_price=low_price,
                        volume=volume,
                    )
            else:
                ticker_price_history_str = f"The price history of {ticker_name} is unknown"
        else:
            ticker_price_history_str = f"The price history of {ticker_name} is unknown"

        return ticker_price_history_str


if __name__ == "__main__":
    finance_2_nlp = finance2nlp("Apple")
    filename = "example_data/finance_description.txt"

    description = finance_2_nlp.finance_company_2_nlp()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(description)
