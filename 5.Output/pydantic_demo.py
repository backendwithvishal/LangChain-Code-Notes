from pydantic import BaseModel, EmailStr , Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Vishal'
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10)
    # email: EmailStr 

new_student = {'age':20, 'cgpa': 9 }

student = Student(**new_student)

print(student)