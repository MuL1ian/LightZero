import numpy as np

v = np.array([True, False, True])
actions_list = ['a', 'b', 'c']


def get_action_mask(v):
    valid_mask = v,
    action_mask = np.zeros(len(actions_list), dtype=np.int8)
    action_mask[np.where(valid_mask)[0]] = 1
    return action_mask

print(get_action_mask(v))