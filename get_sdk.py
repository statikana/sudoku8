from dataclasses import dataclass
from typing import Literal
import numpy as np
import requests


@dataclass
class SDKResponse:
    board: np.ndarray
    solution: np.ndarray
    difficulty: str


def get_sdk(difficulty: Literal["easy", "medium", "hard"]):
    root = "https://youdosudoku.com/api"
    body = {"difficulty": difficulty, "array": True}
    headers = {"Content-Type": "application/json"}

    with requests.post(root, json=body, headers=headers) as response:
        json = response.json()
        return SDKResponse(
            board=np.array(json["puzzle"]).astype(int),
            solution=np.array(json["solution"]).astype(int),
            difficulty=json["difficulty"],
        )
