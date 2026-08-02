from pydantic import BaseModel

class Student(BaseModel):

    name: str

new_student = {'name':'Vishal'}

student = Student(**new_student)

print(type(student))