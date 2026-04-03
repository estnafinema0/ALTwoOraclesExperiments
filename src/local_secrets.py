import pathlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Secrets:
    gigachat_api_key: str
    hf_token: str

    @classmethod
    def load_secrets(cls, root_dir: pathlib.Path) -> 'Secrets':
        with (root_dir / 'secrets.json').open('r') as f:
            data = json.load(f)
        return Secrets(
            data['GIGACHAT_API'],
            data['HF_TOKEN'],
        )
