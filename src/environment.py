#!/usr/bin/env python3
# coding=utf-8
from data.structure import State, Evidence, Action
from typing import Tuple, Set

def get_id_from_evidence(e: Evidence) -> Set[Tuple[str, int]]:
    return set(map(lambda sent: tuple(sent.id), e))

class BaseEnv:
    def __init__(self, K=5):
        self.K = K
    
    def jaccard(self, e1: Evidence, e2: Evidence) -> float:
        sents1 = get_id_from_evidence(e1)
        sents2 = get_id_from_evidence(e2)
        return (len(sents1 & sents2) + 1.0) / (len(sents1 | sents2) + 1.0)

    @classmethod
    def new_state(cls, state: State, action: Action) -> State:
        return State(claim=state.claim,
                     label=state.label,
                     evidence_set=state.evidence_set,
                     candidate=state.candidate + [action.sentence],
                     pred_label=state.pred_label,
                     count=state.count + 1)
    
    def is_done(self, state: State) -> bool:
        return state.count == self.K

    def score(self, state: State) -> float:
        return NotImplementedError()

    def reward(self, state_now: State, state_next: State) -> float:
        return NotImplementedError()

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        return NotImplementedError()


class FeverEnv(BaseEnv):
    def __init__(self, label2id, K=5):
        super(FeverEnv, self).__init__(K)
        self.label2id = label2id

    def reward(self, state: State, action: Action) -> float:
        cond1 = state.label == state.pred_label
        all_evi_id = [get_id_from_evidence(evi) for evi in state.evidence_set]
        if self.is_done(state):
            candidate = get_id_from_evidence(state.candidate)
            cond2 = any([len(ids - candidate) == 0 for ids in all_evi_id])
        else:
            cond2 = any([action.sentence.id in ids for ids in all_evi_id])

        if state.label == self.label2id['NOT ENOUGH INFO']:
            if cond1: return 1.
            elif self.is_done(state): return -1.
            else: return 0.
        else:
            if cond1 and cond2: return 1.
            elif cond1 and not cond2: return 0.
            elif self.is_done(state): return -1.
            else: return 0.

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        state_next = BaseEnv.new_state(state, action)
        done = self.is_done(state_next)
        return state_next, self.reward(state_next, action), done

