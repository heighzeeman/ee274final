"""
Q5 -> The Boy Scout Final Task
"""

from typing import List


def compute_minimal_effort(rope_lengths_arr: List[float]) -> float:
    """
    lengths_arr -> list of rope lengths (positive floating points)
    output -> the value of the minimum effort in joining all the ropes together 
    """
    effort = 0
    while len(rope_lengths_arr) > 1:
        shortest = min(rope_lengths_arr)
        rope_lengths_arr.remove(shortest)
        second_shortest = min(rope_lengths_arr)
        rope_lengths_arr.remove(second_shortest)
        
        rope_lengths_arr.append(shortest + second_shortest)
        effort += shortest + second_shortest
        
    return effort


def test_compute_minimal_effort_func():
    """
    Sample test function.
    please feel free to add more test cases of your choice
    """
    rope_lengths_to_test = [[5, 5], [1, 1, 1, 1], [20, 30, 40]]
    expected_minimal_effort = [10, 8, 140]

    for rope_lengths_arr, expected_effort in zip(rope_lengths_to_test, expected_minimal_effort):
        min_effort = compute_minimal_effort(rope_lengths_arr)
        assert (abs(min_effort - expected_effort) < 1e-6), "incorrect minimum effort"
