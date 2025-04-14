import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDB:
  def __init__(self):
    self.pkl_path = "student_db.pkl"
    self.db = pd.read_pickle(self.pkl_path)
  
  def add_entry(self, entry):
    db_new_entries = pd.DataFrame([entry])
    self.db = pd.concat([self.db, db_new_entries], ignore_index=True)
    self.db.to_pickle(self.pkl_path)
    
  def delete_entry(self, id):
    self.db = self.db[self.db['ID'] != id]
    self.db.to_pickle(self.pkl_path)
    
  def find_similar(self, query_embedding):
    embeddings = np.vstack(self.db['embedding'].values)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    most_similar_row = self.db.iloc[most_similar_idx]
    return most_similar_row

if __name__=="__main__":
  vectordb = VectorDB("student_db.pkl")
  entry = {"ID":1, "name":"san", "embedding":[]}
  new_rows = pd.DataFrame([entry])
  vectordb.db = pd.concat([vectordb.db, new_rows], ignore_index=True)
  print(vectordb.db)
