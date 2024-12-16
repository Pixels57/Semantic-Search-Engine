from typing import Dict, List, Annotated
import numpy as np
import os
import random
from sklearn.cluster import KMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:

    def _load_index(self):
        """
        Loads the centroids and inverted lists from disk.
        """
        if os.path.exists(self.index_path):
            # Load centroids
            self.centroids = np.loadtxt(self.index_path, delimiter=",")

            # Load inverted lists
            inverted_file_path = self.index_path.replace(".csv", "_inverted_lists.txt")
            if os.path.exists(inverted_file_path):
                self.inverted_lists = {}
                with open(inverted_file_path, "r") as f:
                    for line in f:
                        cluster_idx, vector_ids = line.strip().split(":")
                        self.inverted_lists[int(cluster_idx)] = list(map(int, vector_ids.split(",")))
        else:
            raise FileNotFoundError(f"No index file found at {self.index_path}")

    def __init__(self, database_file_path = "saved_db.csv", index_file_path = "index.csv", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.nlist = 10
        self.centroids = None
        self.inverted_lists = None

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self._load_index()
    
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
        # scores = []
        # num_records = self._get_num_records()
        # # here we assume that the row number is the ID of each vector
        # for row_num in range(num_records):
        #     vector = self.get_one_row(row_num)
        #     score = self._cal_score(query, vector)
        #     scores.append((score, row_num))
        # # here we assume that if two rows have the same score, return the lowest ID
        # scores = sorted(scores, reverse=True)[:top_k]
        # return [s[1] for s in scores]

        centroid_scores = []
        for i, centroid in enumerate(self.centroids):
            score = self._cal_score(query, centroid)
            centroid_scores.append((score, i))
        closest_centroids = [x[1] for x in sorted(centroid_scores, reverse=True)[:2]]

        # Step 2: Search within the inverted lists of the closest centroids
        candidates = []
        for centroid_idx in closest_centroids:
            for vector_id in self.inverted_lists.get(centroid_idx, []):
                vector = self.get_one_row(vector_id)
                score = self._cal_score(query, vector)
                candidates.append((score, vector_id))

        # Step 3: Sort candidates by score and return the top-k results
        candidates = sorted(candidates, reverse=True)[:top_k]
        return [x[1] for x in candidates]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    

    #############################################################################

    def _save_index(self):
        # Save centroids to a file
        np.savetxt(self.index_path, self.centroids, delimiter=",")

        # Save inverted lists to a separate file
        inverted_file_path = self.index_path.replace(".csv", "_inverted_lists.txt")
        with open(inverted_file_path, "w") as f:
            for cluster_idx, vector_ids in self.inverted_lists.items():
                f.write(f"{cluster_idx}:{','.join(map(str, vector_ids))}\n")

    def _build_index(self):
        # Placeholder for index building logic
        # root = Node(None, self.get_all_rows())
        # root._build_tree(100, 0.7)

        # print("Hi")

        # all_vectors = self.get_all_rows()
        # num_vectors = all_vectors.shape[0]

        # rng = np.random.default_rng(DB_SEED_NUMBER)
        # self.centroids = rng.random((self.nlist, DIMENSION), dtype=np.float32)

        # for _ in range(10):  # Simple iterative k-means with a fixed number of iterations
        #     assignments = [np.argmin(np.linalg.norm(self.centroids - vec, axis=1)) for vec in all_vectors]
        #     for i in range(self.nlist):
        #         cluster_vectors = all_vectors[np.array(assignments) == i]
        #         if len(cluster_vectors) > 0:
        #             self.centroids[i] = np.mean(cluster_vectors, axis=0)
        # print("Hi")
        # self.inverted_lists = {i: [] for i in range(self.nlist)}
        # for idx, vec in enumerate(all_vectors):
        #     centroid_idx = np.argmin(np.linalg.norm(self.centroids - vec, axis=1))
        #     self.inverted_lists[centroid_idx].append(idx)
        # print("Hi2")

        num_records = self._get_num_records()
        if num_records == 0:
            raise ValueError("The database is empty. Add records before building the index.")
        
        data = self.get_all_rows()

        kmeans = KMeans(n_clusters=self.nlist, random_state=DB_SEED_NUMBER)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_

        cluster_assignments = kmeans.labels_
        self.inverted_lists = {i: [] for i in range(self.nlist)}

        for idx, cluster_idx in enumerate(cluster_assignments):
            self.inverted_lists[cluster_idx].append(idx)

        self._save_index()