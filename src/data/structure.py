#coding: utf8 
from collections import namedtuple
from typing import List, Tuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_actions', 'done'))
Action = namedtuple('Action', ('sentence'))
State = namedtuple('State', ('claim', 'label', 'evidence_set', 'candidate', 'pred_label', 'count'))
Sentence = namedtuple('Sentence', ('id', 'tokens', 'str'))
Claim = namedtuple('Claim', ('id', 'str', 'tokens'))
Evidence = List[Sentence]
EvidenceSet = List[Evidence]
DataSet = List[Tuple[Claim, int, Evidence, List[Sentence]]]

Sentence.__new__.__defaults__ = (None, None)
