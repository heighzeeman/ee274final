import numpy as np
from core.data_block import DataBlock


def biased_coin_generator():
    """
    the function returns H with probability prob_H, and T with prob_T
    """
    # NOTE: hard-coding prob_H to 0.8
    prob_H = 0.8
    prob_T = 1 - prob_H
    # generate random coin toss
    rng = np.random.default_rng()
    toss = rng.choice(["H", "T"], p=[prob_H, prob_T])
    return toss


def shubhams_fair_coin_generator():
    """
    TODO:
    use the biased_coin_generator() function above to implement the fair coin generator

    Outputs:
        toss (str): H, T
        num_biased_tosses (int): number of times the biased_coin_generator() was called to generate the output
    """
    toss = None
    num_biased_tosses = 0

    ##############################
    # ADD CODE HERE
    raise NotImplementedError
    ##############################

    return toss, num_biased_tosses


def test_shubhams_fair_coin_generator():
    """
    TODO:
    write a test to check whether the shubhams_fair_coin_generator() is really generating fair coin tosses

    Also, check if the num_biased_tosses matches the answer which you computed in part 2.
    """

    # perform the experiment
    # feel free to reduce this when you are testing
    num_trials = 10000
    tosses = []
    num_biased_tosses_list = []
    for _ in range(num_trials):
        toss, num_biased_tosses = shubhams_fair_coin_generator()

        # check if the output is indeed in H,T
        assert toss in ["H", "T"]
        tosses.append(toss)
        num_biased_tosses_list.append(num_biased_tosses)

    # NOTE: We are using the DataBlock object from SCL.
    # Take a look at `core/data_block.py` to understand more about the class
    # the get_empirical_distribution() outputs a ProbDist class. Take a look at
    # `core/prob_dist.py` to understand this object better
    tosses_db = DataBlock(tosses)
    empirical_dist_tosses = tosses_db.get_empirical_distribution()

    #############################################
    # ADD CODE HERE
    # 1. add test here to check if empirical_dist_tosses is close to being fair
    # 2. add test to check if avg value of num_biased_tosses matches what you expect
    # (both within a reasonable error range)
    # You can use np.testing.assert_almost_equal() to check if two numbers are close to some error margin
    raise NotImplementedError
    #############################################
