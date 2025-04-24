from typing import Union
import datetime as dt


class Time:
    def __init__(self, date_start: str):
        self._date_format = "%Y-%m-%d %H:%M:%S"
        self._date_start = date_start
        self._periods_per_day = 24

    def __str2datetime(self, datetime_str: str, format: str):
        datetime = datetime_str
        if isinstance(datetime_str, str):
            datetime = dt.datetime.strptime(datetime_str, format)
        return datetime

    def __check_datetime_str(self, tag, format):
        if isinstance(tag, str):
            try:
                dt.datetime.strptime(tag, format)
                return True
            except:
                return False
        else:
            return False

    def __hour2period(
        self, period: Union[int, float] = 1.0, return_float: bool = False
    ) -> Union[int, float]:
        period = period * self._periods_per_day / 24
        if return_float:
            return period
        return round(period)

    def __datetime2period(
            self,
            datetime: Union[dt.datetime, str],
            start_datetime: Union[dt.datetime, str] = None,
    ) -> Union[int, float]:
        datetime = self.__str2datetime(datetime, self._date_format)
        start_datetime = (
            self.__str2datetime(start_datetime, self._date_format) if start_datetime else self._date_start
        )
        period = int((datetime - start_datetime).total_seconds() / 60 / 60)
        return self.__hour2period(period)

    def __period2datetime(self, period: int):
        return self._date_start + dt.timedelta(
            hours=(period % self._periods_per_day) * (24 // self._periods_per_day),
            days=period // self._periods_per_day,
        )

    def __replace_date(self, tag: Union[int, float, str, list, dict]) -> None:
        if isinstance(tag, dict):
            for key, value in tag.copy().items():
                if self.__check_datetime_str(key, self._date_format):
                    del tag[key]
                    new_key = self.__datetime2period(key)
                    tag[new_key] = value
                else:
                    pass
                if self.__check_datetime_str(value, self._date_format):
                    new_value = self.__datetime2period(value)
                    tag[key] = new_value
                else:
                    self.__replace_date(value)
        elif isinstance(tag, list):
            for value in tag.copy():
                if self.__check_datetime_str(value, self._date_format):
                    tag.remove(value)
                    new_value = self.__datetime2period(value)
                    tag.append(new_value)
                else:
                    self.__replace_date(value)
        else:
            return

    def transition_to_periods(self, input_data: dict) -> dict:
        self.__replace_date(input_data)
        return input_data