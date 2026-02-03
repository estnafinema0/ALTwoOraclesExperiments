from abc import ABC, abstractmethod
from typing import Self
import os

class Stringifiable(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    @staticmethod
    def from_str(s: str) -> Self:
        pass

class Dumpable(ABC):
    @abstractmethod
    def dump(self, root_dir: os.PathLike, filename: str | None = None): # TODO:  do not rewrite 
        pass

    @abstractmethod
    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> Self:
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def get_config(self, external_id: str | None = None) -> dict:
        pass

    @abstractmethod
    def get_config_filename(self, external_id: str | None = None) -> str:
        pass

class JSONifiable(ABC):
    @abstractmethod
    def to_json(self) -> dict:
        pass

    @abstractmethod
    @staticmethod
    def from_json(data: dict) -> Self:
        pass