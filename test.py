import numpy as np
from vec_db import VecDB

db = VecDB(db_size = 20*(10**6))
# query_vector = np.random.rand(1,70) # Query vector of dimension 70
# similar_images = db.retrieve(query_vector, top_k=5)
# print(similar_images)