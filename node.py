from typing import List, Optional
import random

import numpy as np


class Node(object):
    """Initialize with a set of vectors, then call `split()`.
    """

    def __init__(self, ref: np.ndarray, vecs: List[np.ndarray]):
        self._ref = ref
        self._vecs = vecs
        self._left = None
        self._right = None

    @property
    def ref(self) -> Optional[np.ndarray]:
        """Reference point in n-d hyperspace. Evaluates to `False` if root node.
        """
        return self._ref

    @property
    def vecs(self) -> List[np.ndarray]:
        """Vectors for this leaf node. Evaluates to `False` if not a leaf.
        """
        return self._vecs

    @property
    def left(self) -> Optional[object]:
        """Left node.
        """
        return self._left

    @property
    def right(self) -> Optional[object]:
        """Right node.
        """
        return self._right

    def split(self, K: int, imb: float) -> bool:
    # stopping condition: maximum # of vectors for a leaf node
        if len(self._vecs) <= K:
            return False

        # continue for a maximum of 5 iterations
        for n in range(5):
            left_vecs = []
            right_vecs = []

            # Shuffle the vectors and select two random references
            np.random.shuffle(self._vecs)  # Shuffle the list of vectors
            left_ref = self._vecs[0]  # First vector after shuffle
            right_ref = self._vecs[1]  # Second vector after shuffle

            # Remove these references from the list (to avoid reusing them)
            remaining_vecs = self._vecs[2:]

            # Split vectors into halves based on proximity to the references
            for vec in remaining_vecs:
                dist_l = np.linalg.norm(vec - left_ref)
                dist_r = np.linalg.norm(vec - right_ref)
                if dist_l < dist_r:
                    left_vecs.append(vec)
                else:
                    right_vecs.append(vec)

            # Check to make sure that the tree is mostly balanced
            r = len(left_vecs) / len(remaining_vecs)
            if r < imb and r > (1 - imb):
                self._left = Node(left_ref, left_vecs)
                self._right = Node(right_ref, right_vecs)
                return True

            # Redo tree build process if imbalance is high
            self._vecs = [left_ref, right_ref] + remaining_vecs

        return False



def _select_nearby(node: Node, q: np.ndarray, thresh: int = 0):
    """Functions identically to _is_query_in_left_half, but can return both.
    """
    if not node.left or not node.right:
        return ()
    dist_l = np.linalg.norm(q - node.left.ref)
    dist_r = np.linalg.norm(q - node.right.ref)
    if np.abs(dist_l - dist_r) < thresh:
        return (node.left, node.right)
    if dist_l < dist_r:
        return (node.left,)
    return (node.right,)


def _build_tree(node, K: int, imb: float):
    """Recurses on left and right halves to build a tree.
    """
    node.split(K=K, imb=imb)
    if node.left:
        _build_tree(node.left, K=K, imb=imb)
    if node.right:
        _build_tree(node.right, K=K, imb=imb)


def build_forest(vecs: List[np.ndarray], N: int = 32, K: int = 64, imb: float = 0.95) -> List[Node]:
    """Builds a forest of `N` trees.
    """
    forest = []
    for _ in range(N):
        root = Node(None, vecs)
        _build_tree(root, K, imb)
        forest.append(root)
    return forest


def _query_linear(vecs: List[np.ndarray], q: np.ndarray, k: int) -> List[np.ndarray]:
    return sorted(vecs, key=lambda v: np.linalg.norm(q-v))[:k]


def _query_tree(root: Node, q: np.ndarray, k: int) -> List[np.ndarray]:
    """Queries a single tree.
    """

    pq = [root]
    nns = []
    while pq:
        node = pq.pop(0)
        nearby = _select_nearby(node, q, thresh=0.05)

        # if `_select_nearby` does not return either node, then we are at a leaf
        if nearby:
            pq.extend(nearby)
        else:
            nns.extend(node.vecs)

    # brute-force search the nearest neighbors
    return _query_linear(nns, q, k)


def query_forest(forest: List[Node], q, k: int = 10):
    nns = set()
    for root in forest:
        # merge `nns` with query result
        res = _query_tree(root, q, k)
        
        # Convert numpy arrays to tuples for hashability before adding to the set
        nns.update(tuple(r) for r in res)
    
    return _query_linear(list(nns), q, k)

