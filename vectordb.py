import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDB:
  def __init__(self, pkl_path):
    self.db = pd.read_pickle(pkl_path)
  
  def add_entry(self, entry):
    db_new_entries = pd.DataFrame(entry)
    self.db = pd.concat([self.db, db_new_entries])

  def delete_entry(self, id):
    self.db = self.db[self.db['ID'] != id]
    
  def find_similar(self, query_embedding):
    embeddings = np.vstack(self.db['embedding'].values)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    most_similar_row = self.db.iloc[most_similar_idx]
    return most_similar_row
