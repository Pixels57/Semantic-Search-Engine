import numpy as np
from typing import List, Optional

class Node(object):


    def __init__(self, ref: np.ndarray, vecs: List[np.ndarray]) -> None:
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
    
    def _is_query_in_left_half(q, node):
        # returns `True` if query vector resides in left half
        dist_l = np.linalg.norm(q - node.vecs[0])
        dist_r = np.linalg.norm(q - node.vecs[1])
        return dist_l < dist_r
    
    def split_node(self, K: int, imb: float) -> bool:

        # stopping condition: maximum # of vectors for a leaf node
        if len(self._vecs) <= K:
            return False

        # continue for a maximum of 5 iterations
        for n in range(5):
            left_vecs = []
            right_vecs = []

            # take two random indexes and set as left and right halves
            left_ref = self._vecs.pop(np.random.randint(len(self._vecs)))
            right_ref = self._vecs.pop(np.random.randint(len(self._vecs)))

            # split vectors into halves
            for vec in self._vecs:
                dist_l = np.linalg.norm(vec - left_ref)
                dist_r = np.linalg.norm(vec - right_ref)
                if dist_l < dist_r:
                    left_vecs.append(vec)
                else:
                    right_vecs.append(vec)

            # check to make sure that the tree is mostly balanced
            r = len(left_vecs) / len(self._vecs)
            if r < imb and r > (1 - imb):
                self._left = Node(left_ref, left_vecs)
                self._right = Node(right_ref, right_vecs)
                return True

            # redo tree build process if imbalance is high
            self._vecs.append(left_ref)
            self._vecs.append(right_ref)

        return False
        
    #Recurses on left and right halves to build a tree.
    def _build_tree(self, K: int, imb: float):
        self.split_node(K=K, imb=imb)
        if self.left:
            self._build_tree(self.left, K=K, imb=imb)
        if self.right:
            self._build_tree(self.right, K=K, imb=imb)
    