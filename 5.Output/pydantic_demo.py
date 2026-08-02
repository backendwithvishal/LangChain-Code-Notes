from pydantic import BaseModel, EmailStr , Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Vishal'
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10, default = 8,description='A decimal value representing the cgpa of the student')
    # email: EmailStr 

new_student = {'age':20}

student = Student(**new_student)

print(student)

student_json = student.model_dump_json()