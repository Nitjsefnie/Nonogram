from solver import *


def test_easy(drawing=False) -> None:
    solve_folder("demo_nonograms/easy/", drawing)


def test_medium(drawing=False) -> None:
    solve_folder("demo_nonograms/medium/", drawing)


def test_pika(drawing=False, cheated_pixels=[]) -> None:
    solve_file('demo_nonograms/impossible/pikachu', drawing, cheated_pixels)


def test_hard(drawing=False) -> None:
    solve_folder("demo_nonograms/hard/")
