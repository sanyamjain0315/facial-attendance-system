from datetime import datetime
import pandas as pd

class AttendanceLogger:
  def __init__(self):
    self.pkl_path = "data/attendance.pkl"
    self.db = pd.read_pickle(self.pkl_path)
  
  def log_entry(self, student_info):
    entry = {
      "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "ID":student_info['ID'],
      "name": student_info["name"]
    }
    db_new_entries = pd.DataFrame([entry])
    self.db = pd.concat([self.db, db_new_entries], ignore_index=True)
    self.db.to_pickle(self.pkl_path)

  def clear_logs(self):
    self.db = self.db[0:0]
    self.db.to_pickle(self.pkl_path)

if __name__=="__main__":
  al = AttendanceLogger()
  al.clear_logs()
