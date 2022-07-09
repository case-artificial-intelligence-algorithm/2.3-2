#!/usr/bin/env python3

from my_solution import solution


# 测试用例
def test_solution():

    # 正确答案
    correct_solution = [(0.13618343634378283, 0.5874928452176532), (0.36042649506503965, 0.6238096394561127),
     (0.05245740880129763, 0.6725707311544702), (0.14784234630593573, 0.8614284493193792), 
     (0.13514767775928418, 0.7249998095232654)]
    # 程序求解结果
    result = solution()
    assert correct_solution == result

