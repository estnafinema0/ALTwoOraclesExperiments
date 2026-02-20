from abc import ABC, abstractmethod, ABCMeta
import enum
from typing import Self
import pathlib
import dataclasses


def open_subbuild(*parts) -> pathlib.Path:
    path = pathlib.Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Constant:
    def __init__(self, value):
        self.value = value

    def __get__(self, *args):
        return self.value

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __getitem__(self, key):
        return self.value[key]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'


class EnumABCMeta(ABCMeta, enum.EnumMeta):
    pass
