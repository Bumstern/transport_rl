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

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.__list_by_id[index]
        elif isinstance(index, str):
            return self.__named_dict[index]
        elif isinstance(index, float):
            if index == int(index):
                return self.__list_by_id[int(index)]
            else:
                raise ValueError('Float index')
        else:
            raise NotImplementedError('Unsupported type')

    def __iter__(self):
        self.__remaining_requests = deque(self.__list_by_id)
        return self

    def __next__(self):
        try:
            return self.__remaining_requests.popleft()
        except:
            raise StopIteration

    def __len__(self):
        return len(self.__list_by_id)