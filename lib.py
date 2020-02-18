import numpy as np


def read_machine_detail(filename):
    with open(filename) as f:
        lines = f.readlines()
    m_count = len(lines[0].split())
    d_count = len(lines)
    matrix = np.zeros((d_count, m_count), dtype=int)
    for i, line in enumerate(lines):
        matrix[i] = [int(x) for x in line.split()]
    return matrix


def read_detail_tree(filename, d_count):
    with open(filename) as f:
        lines = f.readlines()

    matrix = np.zeros((d_count, d_count), dtype=int)
    for line in lines:
        dleft, _, drights = line.partition(":")
        dleft = int(dleft[1:])
        for x in drights.split():
            cnt, _, dright = x.partition("d")
            cnt = int(cnt)
            dright = int(dright)
            matrix[dleft - 1][dright - 1] = cnt

    return matrix
