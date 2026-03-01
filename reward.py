import numpy as np
MAX_FATIGUE = 10
OPERATIONS = 5
k = 0.3 
ACTIONS = ['Safe', 'Fast']
def reward(action, fail=False):
    if fail:
        return -10
    return 10 - k * (5 if action == 'Safe' else 2)
def transitions(fatigue, prev_fast, action):
    result = []
    if action == 'Safe':
        for inc, prob in [(1, 0.8), (2, 0.2)]:
            F_next = min(fatigue + inc, MAX_FATIGUE)
            fail = F_next >= MAX_FATIGUE
            result.append((F_next, 0, prob, fail))
    else:
        if prev_fast == 0: 
            for inc, prob in [(3, 0.7), (4, 0.3)]:
                F_next = fatigue + inc
                if F_next >= 8:
                    F_tear = F_next + 4
                    fail_tear = F_tear >= MAX_FATIGUE
                    result.append((min(F_tear, MAX_FATIGUE), 1, prob*0.2, fail_tear))
                    result.append((min(F_next, MAX_FATIGUE), 1, prob*0.8, F_next >= MAX_FATIGUE))
                else:
                    result.append((min(F_next, MAX_FATIGUE), 1, prob, F_next >= MAX_FATIGUE))
        else:
            for inc, prob in [(5, 0.6), (7, 0.4)]:
                F_next = fatigue + inc
                if F_next >= 8:
                    F_tear = F_next + 4
                    fail_tear = F_tear >= MAX_FATIGUE
                    result.append((min(F_tear, MAX_FATIGUE), 1, prob*0.2, fail_tear))
                    result.append((min(F_next, MAX_FATIGUE), 1, prob*0.8, F_next >= MAX_FATIGUE))
                else:
                    result.append((min(F_next, MAX_FATIGUE), 1, prob, F_next >= MAX_FATIGUE))
    return result
V = np.zeros((OPERATIONS+1, MAX_FATIGUE+1, 2))
policy = np.empty((OPERATIONS+1, MAX_FATIGUE+1, 2), dtype=object)
for ops_left in range(1, OPERATIONS+1):
    for F in range(MAX_FATIGUE+1):
        for prev in [0, 1]:
            best_val = -np.inf
            best_action = None
            for action in ACTIONS:
                val = 0
                for F_next, prev_next, prob, fail in transitions(F, prev, action):
                    r = reward(action, fail)
                    if ops_left > 1 and not fail:
                        r += V[ops_left-1][F_next][prev_next]
                    val += prob * r
                if val > best_val:
                    best_val = val
                    best_action = action
            V[ops_left][F][prev] = best_val
            policy[ops_left][F][prev] = best_action