import pandas as pd
import numpy as np
import streamlit as st
from vectordb import VectorDB
from facial_recognizer import FacialRecognizer
import cv2
import secrets


# Loading pre-requisites
student_db = pd.read_pickle("student_db.pkl")
attendance = pd.read_pickle("attendance.pkl")
vectordb = VectorDB("student_db.pkl")
fc = FacialRecognizer()

# --- Functions for Student Actions ---
def log_attendance():
    st.header("Log Attendance")
    st.write("Upload an image or use your camera to log attendance.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Placeholder: Process the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("Log Attendance from Upload"):
                # Placeholder for facial recognition & logging logic
                st.success("Attendance logged from uploaded image!")

    with col2:
        camera_image = st.camera_input("Capture Image")
        if camera_image is not None:
            # Placeholder: Process the camera image
            st.image(camera_image, caption="Captured Image", use_container_width=True)
            if st.button("Log Attendance from Camera"):
                # Placeholder for facial recognition & logging logic
                st.success("Attendance logged from camera!")
    
def view_attendance_history():
    st.header("Attendance History")
    st.write("Display previous attendance records here.")
    # Placeholder: In a real app, retrieve and display attendance history from a database.
    st.write("Attendance records would be listed here.")

# --- Functions for Teacher Actions ---
def add_student():
    st.header("Add Student")
    st.write("Enter new student details:")
    student_name = st.text_input("Student Name")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = fc.extract_boxes(image)
    if len(boxes) > 1:
        faces = fc.get_cropped_faces(image, boxes)
        box = _select_boxes(faces, boxes)
    if st.button("Add Student"):
        embedding = fc.calculate_embeddings(image, [box])[0]
        student_id = secrets.token_hex(4)
        vectordb.add_entry({
            "ID":student_id,
            "name":student_name,
            "embedding":embedding,
        })
        st.success(f"Student {student_name} (ID: {student_id}) added successfully!")
        
def _select_boxes(faces, boxes, images_per_row=3):
    st.write("Please Select only one face")
    cols = st.columns(len(faces))
    for i, face in enumerate(faces):
        with cols[i]:
            st.write(f"Person {i+1}")
            st.image(face, use_container_width=True)
    chosen_option = st.radio("Select person",range(1, len(faces)+1), index=None)
    if chosen_option == None:
        st.stop()
    else:
        st.write("You chose:", chosen_option)
        return boxes[chosen_option - 1]

def delete_student():
    st.header("Delete Student")
    students = vectordb.db

    if len(students) == 0:
        st.warning("No students available to delete.")
        return

    # Display the current list
    st.dataframe(students[["ID", "name"]])

    # Create display names for selection
    student_choices = students.apply(lambda row: f"{row['name']} (ID: {row['ID']})", axis=1)
    selected_student = st.selectbox("Select a student to delete:", student_choices)

    if selected_student:
        # Extract the student ID from the selected text
        selected_id = selected_student.split("ID: ")[1].rstrip(")")

        if st.button("Delete Selected Student"):
            vectordb.delete_entry(selected_id)
            st.success(f"Student {selected_student} has been deleted!")

def view_attendance_records():
    st.header("Attendance Records")
    st.write("Display attendance records for students here.")
    # Placeholder: Retrieve and display attendance records from the database
    st.write("Attendance records would appear here.")

def view_all_students():
    st.header("All Registered Students")
    students = vectordb.db
    print(vectordb.db)
    if len(students) == 0:
        st.warning("No students found in the database.")
        return
    st.dataframe(students[["ID", "name"]])  # Only show ID and name

# --- Main Application Flow ---
def main():
    st.title("Facial Attendance System")
    st.write("Welcome to the Facial Attendance System. Please choose your role to continue.")

    # Role selection: student or teacher
    role = st.radio("Select your role:", ("Student", "Teacher"))

    if role == "Student":
        st.subheader("Student Dashboard")
        option = st.selectbox("Choose an option:", ("Log Attendance", "View Attendance History"))
        if option == "Log Attendance":
            log_attendance()
        elif option == "View Attendance History":
            view_attendance_history()
    elif role == "Teacher":
        st.subheader("Teacher Dashboard")
        option = st.selectbox("Choose an option:", (
            "Add Student", 
            "Delete Student", 
            "View Attendance Records", 
            "View All Students"
        ))
        if option == "Add Student":
            add_student()
        elif option == "Delete Student":
            delete_student()
        elif option == "View Attendance Records":
            view_attendance_records()
        elif option == "View All Students":
            view_all_students()

if __name__ == '__main__':
    main()
