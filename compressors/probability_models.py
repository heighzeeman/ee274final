"""This file contains probability models such as adaptive kth order etc. to
be used with arithmetic/range coders.

"""
from __future__ import annotations # Useful to allow TreeNode type hinting to work without some ugly workarounds.
import abc
import copy
from typing import List, Tuple, Optional

import numpy as np
import numexpr as ne

from core.prob_dist import Frequencies


class FreqModelBase(abc.ABC):
    """Base Freq Model

    The Arithmetic Entropy Coding (AEC) encoder can be thought of consisting of two parts:
    1. The probability model
    2. The "lossless coding" algorithm which uses these probabilities

    Note that the probabilities/frequencies coming from the probability model are fixed in the simplest Arithmetic coding
    version, but they can be modified as we parse each symbol.
    This class represents a generic "probability Model", but using frequencies (or counts), and hence the name FreqModel.
    Frequencies are used, mainly because floating point values can be unpredictable/uncertain on different platforms.

    Some typical examples of Freq models are:

    a) FixedFreqModel -> the probability model is fixed to the initially provided one and does not change
    b) AdaptiveIIDFreqModel -> starts with some initial probability distribution provided
        (the initial distribution is typically uniform)
        The Adaptive Model then updates the model based on counts of the symbols it sees.

    Args:
        freq_initial -> the frequencies used to initialize the model
        max_allowed_total_freq -> to limit the total_freq values of the frequency model
    """

    def __init__(self, freqs_initial: Frequencies, max_allowed_total_freq):
        # initialize the current frequencies using the initial freq.
        # NOTE: the deepcopy here is needed as we modify the frequency table internally
        # so, if it is used elsewhere externally, then it can cause unexpected issued
        self.freqs_current = copy.deepcopy(freqs_initial)
        self.max_allowed_total_freq = max_allowed_total_freq

    @abc.abstractmethod
    def update_model(self, s):
        """updates self.freqs

        Takes in as input the next symbol s and updates the
        probability distribution self.freqs (represented in terms of frequencies)
        appropriately. See examples below.
        """
        raise NotImplementedError  # update the probability model here


class FixedFreqModel(FreqModelBase):
    def update_model(self, s):
        """function to update the probability model

        In this case, we don't do anything as the freq model is fixed

        Args:
            s (Symbol): the next symbol
        """
        # nothing to do here as the freqs are always fixed
        pass


class AdaptiveIIDFreqModel(FreqModelBase):
    def update_model(self, s):
        """function to update the probability model

        - We start with uniform distribution on all symbols
        ```
        Freq = [A:1,B:1,C:1,D:1] for example.
        ```
        - Every time we see a symbol, we update the freq count by 1
        - Arithmetic coder requires the `total_freq` to remain below a certain value
        If the total_freq goes beyond, then we divide all freq by 2 (keeping minimum freq to 1)

        Args:
            s (Symbol): the next symbol
        """
        # updates the model based on the next symbol
        self.freqs_current.freq_dict[s] += 1

        # if total_freq goes beyond a certain value, divide by 2
        # NOTE: there can be different strategies here
        if self.freqs_current.total_freq >= self.max_allowed_total_freq:
            for s, f in self.freqs_current.freq_dict.items():
                self.freqs_current.freq_dict[s] = max(f // 2, 1)


class AdaptiveOrderKFreqModel(FreqModelBase):
    """kth order adaptive frequency model.

    Parameters:
        alphabet: the alphabet (provided as a list)
        k:        the order, k >= 0 (kth order means we use past k to predict next, k=0 means iid)

    """

    def __init__(self, alphabet: List, k: int, max_allowed_total_freq):
        assert k >= 0
        self.k = k
        # map alphabet to index from 0 to len(alphabet) so we can use with numpy array
        self.alphabet_to_idx = {alphabet[i]: i for i in range(len(alphabet))}
        # keep freq/counts of (k+1) tuples, initialize with all 1s (uniform)
        self.freqs_kplus1_tuple = np.ones([len(alphabet)] * (k + 1), dtype=int)
        self.max_allowed_total_freq = max_allowed_total_freq
        # keep track of past k symbols (i.e., alphabet index) seen. Initialize with all 0s.
        # Note that all zeros refers to the first element in the alphabet list. This is an
        # arbitrary choice made to simplify later processing rather than doing special case
        # for the first few symbols
        self.past_k = [0] * k

        self.alphabet = alphabet

    @property
    def freqs_current(self):
        """Calculate the current freqs. For order 0, we just give back the freqs. For k > 0,
        we use the past k symbols to pick out the corresponding frequencies for the (k+1)th.
        """
        if self.k > 0:
            # convert self.past_k to enable indexing
            # use np.ravel to convert to flat array
            freqs_given_context = np.ravel(self.freqs_kplus1_tuple[tuple(self.past_k)])
        else:
            freqs_given_context = self.freqs_kplus1_tuple
        # convert from list of frequencies to Frequencies object
        return Frequencies(dict(zip(self.alphabet, freqs_given_context)))

    def update_model(self, s):
        """function to update the probability model. This basically involves update the count
        for the most recently seen (k+1) tuple.

        - Arithmetic coder requires the `total_freq` to remain below a certain value
        If the total_freq goes beyond, then we divide all freq by 2 (keeping minimum freq to 1)

        Args:
            s (Symbol): the next symbol
        """
        # updates the model based on the new symbol
        # index self.freqs_kplus1_tuple using (past_k, s) [need to map s to index]
        current_tuple = (*self.past_k, self.alphabet_to_idx[s])
        self.freqs_kplus1_tuple[current_tuple] += 1

        # if k > 0, update past_k list
        if self.k > 0:
            self.past_k = self.past_k[1:] + [self.alphabet_to_idx[s]]

        # if total_freq goes beyond a certain value, divide by 2
        # NOTE: there can be different strategies here
        # NOTE: we only need the frequencies for each (k+1) tuple to
        # sum to less than max_allowed_total_freq
        if np.sum(self.freqs_kplus1_tuple[current_tuple]) >= self.max_allowed_total_freq:
            self.freqs_kplus1_tuple[current_tuple] = np.max(self.freqs_kplus1_tuple[current_tuple] // 2, 1)



class TreeNode():
    """Tree node for use in CTW.
    """
    lprob: float = 0.0
    lktp: float = 0.0
    a: int = 0
    b: int = 0
    left: Optional[TreeNode] = None
    right: Optional[TreeNode] = None
    
    @property 
    def isLeaf(self) -> bool:
        return self.left is None and self.right is None
        
    @property
    def snapshot(self) -> Tuple[float, float, int, int]:
        """Returns a copy of the data currently attached to a TreeNode
        """
        return (self.lprob, self.lktp, self.a, self.b)
    
    def KTupdate(self, bit):
        """Updates the KT estimator at the current node.
        """
        if bit == 0:
            self.lktp += np.log2(self.a*2 + 1) - 1 - np.log2(self.a + self.b + 1)
            self.a += 1
        elif bit == 1:
            self.lktp += np.log2(self.b*2 + 1) - 1 - np.log2(self.a + self.b + 1)
            self.b += 1
        else:
            raise RuntimeError('KTupdate called with symbol = ' + str(bit))
            
    def CTWupdate(self):
        if self.isLeaf:
            self.lprob = self.lktp
        else:
            if self.left is not None:
                leftlp = self.left.lprob
            else:
                leftlp = 0.0 # Assigned probability 1 if unvisited
            if self.right is not None:
                rightlp = self.right.lprob
            else:
                rightlp = 0.0 # Assigned probability 1 if unvisited
            self.lprob = np.logaddexp2( self.lktp, leftlp + rightlp ) - 1


class ContextTreeWeightingKFreqModel(FreqModelBase):
    """Depth k adaptive CTW frequency model.

    Parameters:
        alphabet:   the alphabet (provided as a list). Has to be a binary alphabet (only two symbols).
        k:          the depth, k >= 0 (kth order means we use up to k depth in our context tree model, i.e. up to 
        past k to predict next symbol, k=0 means we assume an iid source that we estimate with a single KT 
        estimator).
        past_k:     the initial history ("past k" symbols ) to be used for the start of the sequence. If this is not 
        provided, this is arbitrarily set to k starting 0's..
    """
    k: int
    past_k: List[int]
    max_allowed_total_freq: int
    alphabet: List[str]
    root: TreeNode

    def __init__(self, alphabet, k, max_allowed_total_freq, past_k = None):
        assert k >= 0
        assert len(alphabet) == 2
        self.k = k
        # map alphabet to index from 0 to len(alphabet) for easy use with numpy arrays/lists
        self.alphabet_to_idx = {alphabet[i]: i for i in range(len(alphabet))}        
        self.max_allowed_total_freq = max_allowed_total_freq
        
        if past_k is None:
        # Arbitrarily set history to k 0's
            self.past_k = [0] * k
        else:
        # Use provided history
            assert len(past_k) == k
            self.past_k = [self.alphabet_to_idx[s] for s in past_k]
        self.alphabet = alphabet
        self.root = TreeNode()
        # For internal use in performance optimization
        self._future = False # Indicates if the context tree is "in the future" by already predicting and updating 0 for the next symbol
        self._prev_traverse = []


    @property
    def freqs_current(self):
        """ function to return Pr[0|history] and Pr[1|history]
        
        Pr[0|history] = Pr[history followed by 0] / Pr[history]
        Pr[history] is what is usually stored in the root node of the tree.
        The canonical way to calculate Pr[history followed by 0] is to "fake" an update to the tree by observing
        a 0, then reverting the update. We defer the reversions by noting that if the next symbol is indeed 0,
        no reversion or further action is necessary. Otherwise, we can revert them and update as per usual.
        """
        if self._future:
        # The current tree is "in the future": we already have Pr[history followed by 0] in the root node.
            new_log_prob = self.root.lprob
            self.revert(self._prev_traverse)
            self._future = False
            prob_zero = np.exp2(new_log_prob - self.root.lprob)
        else:
            old_log_prob = self.root.lprob
            self._prev_traverse = self.traverse(self.past_k, 0)
            self._future = True
            prob_zero = np.exp2(self.root.lprob - old_log_prob)
            
        probabilities = np.array( [ prob_zero, 1.0 - prob_zero ] )
        frequencies = np.around((self.max_allowed_total_freq >> 2) * probabilities).astype(int)
    
        return Frequencies(dict(zip(self.alphabet, frequencies)))


    def update_model(self, s):
        """function to update the probability model. We update the probabilities corresponding to each of the k suffixes self.past_k[-1:] to self.past_k[-k:], as well as the empty suffix, by traversing the context tree.

        Args:
            s (Symbol): the next symbol
        """
        idx = self.alphabet_to_idx[s]
        
        if not self._future:
            self.traverse(self.past_k, idx)
        elif idx != 0:
            # The tree is in the future and has calculated Pr[history followed by 0].
            # We should calculate Pr[history followed by s] instead. So, revert the 0 observation and update with s.
            self.revert(self._prev_traverse)
            self.traverse(self.past_k, idx)
        
        self._future = False

        # if k > 0, update past_k list
        if self.k > 0:
            self.past_k = self.past_k[1:] + [ idx ]
    
    
    def traverse(self, past_k, idx):
        """helper function that performs all the context tree update logic.

        Args:
            idx: index of the next symbol observed.
        """
        traversed = [] # Contains the nodes that we visit, as well as their pre-update data.
        curr_node = self.root
        traversed.append( (0, curr_node.snapshot, curr_node) )
        curr_node.KTupdate(idx)
        
        # Traverse down the tree and perform the relevant KT estimation updates
        for i in range(1, self.k + 1):
            mode = 0
            if past_k[-i] == 0:
                if curr_node.left is None:
                    curr_node.left = TreeNode()
                    mode = 1 # Signifies that a new treenode was added as the left child
                curr_node = curr_node.left
            elif past_k[-i] == 1:
                if curr_node.right is None:
                    curr_node.right = TreeNode()
                    mode = 2 # Signifies that a new treenode was added as the right child
                curr_node = curr_node.right
            else:
                raise RuntimeError(f'past_k[-{str(i)}] contains a non-binary index {str(past_k[-i])}')
            
            traversed.append( (mode, curr_node.snapshot, curr_node) )
            curr_node.KTupdate(idx)
        
        # Now update the CTW probabilities bottom-up, recalling the recursive definition from the leaves.
        for i in range(1, len(traversed) + 1):
            node = traversed[-i][2]
            node.CTWupdate()
                
        return traversed
        
        
    def revert(self, traversed):
        """helper function that reverts a context tree, given its history.
           This does not really need to be an instance or even class method, but is included for organizational purposes.
           
        Args:
            traversed: the history of the context tree
        """
        for i, (mode, snapshot, node) in enumerate(traversed):
            if mode == 0:
                node.lprob, node.lktp, node.a, node.b = snapshot
            # Below cases: this and subsequent nodes must be new, so we do not need to revert them
            elif mode == 1:
                traversed[i-1][2].left = None
                break 
            else: # mode == 2
                traversed[i-1][2].right = None
                break
                