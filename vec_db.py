from typing import Dict, List, Annotated
import numpy as np
import os
import random
from node import Node
from node import _select_nearby
from node import _build_tree
from node import build_forest
from node import _query_linear
from node import _query_tree
from node import query_forest
import pickle

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        self.forest = self.load_index()
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        if self.forest is None:
            raise ValueError("Index is not loaded. Please load the index.")
        
        # Use the pre-loaded index (forest) to retrieve nearest neighbors
        nearest_neighbors = query_forest(self.forest, query, top_k)
        return nearest_neighbors
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    

    #############################################################################
    def _build_index(self):
        # Build the index starting from the root node
        self.root = Node(None, self.get_all_rows())
        # _build_tree(self.root, K=100, imb=0.95)

        # Create a forest of trees
        self.forest = build_forest(self.get_all_rows(), N=1, K=200, imb=0.95)

        # Save the index to a file
        with open(self.index_path, "wb") as f:
            pickle.dump(self.forest, f)
        print(f"Index saved to {self.index_path}")

    def load_index(self):
        # Load the index from the file
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                forest = pickle.load(f)
            print(f"Index loaded from {self.index_path}")
            return forest
        else:
            print("Index not found. Please build the index first.")
            return None


