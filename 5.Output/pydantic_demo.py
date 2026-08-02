# Import BaseModel, Field, and EmailStr from Pydantic for data validation and schema definition
from pydantic import BaseModel, EmailStr, Field
# Import Optional for fields that can be None
from typing import Optional

# Define a Pydantic data model for Student with field validation rules
class Student(BaseModel):
    name: str = 'Vishal'  # Default name value
    age: Optional[int] = None  # Optional integer field, defaults to None
    cgpa: float = Field(gt=0, lt=10, default=8, description='A decimal value representing the cgpa of the student')  # Value must be between 0 and 10
    # email: EmailStr  # Optional email field with format validation

# Dictionary containing sample input data
new_student = {'age': 20}

# Create and validate a Student object using dictionary unpacking
student = Student(**new_student)

# Print the created Student object
print(student)

# Convert the Pydantic Student instance into a JSON string
student_json = student.model_dump_json()