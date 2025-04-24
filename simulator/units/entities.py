from collections import deque
from functools import singledispatchmethod
from typing import Type, overload

from pydantic import ValidationError


class Entities:

    def __init__(self, data: list[dict], EntityClass: Type):
        self.__named_dict = {}
        self.__list_by_id = []

        try:
            for elem in data:
                elem["id"] = len(self.__list_by_id)
                new_entity = EntityClass(**elem)
                self.__list_by_id.append(new_entity)
                self.__named_dict[new_entity.info.name] = new_entity
        except ValidationError as e:
            print(e.json())
            raise

    @singledispatchmethod
    def __getitem__(self, index):
        raise NotImplementedError('Unsupported type')

    @__getitem__.register
    def __getitem__(self, index: int):
        return self.__list_by_id[index]

    @__getitem__.register
    def _(self, index: str):
        return self.__named_dict[index]

    def __iter__(self):
        self.__remaining_requests = deque(self.__list_by_id)
        return self

    def __next__(self):
        try:
            return self.__remaining_requests.popleft()
        except:
            raise StopIteration
