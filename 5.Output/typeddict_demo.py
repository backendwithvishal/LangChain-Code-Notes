# Import TypedDict from the typing module for dictionary type hinting
from typing import TypedDict

# Define a TypedDict schema to specify expected keys and their data types
class Person(TypedDict):
    name: str
    age: int

# Create a dictionary following the Person schema structure
# Note: TypedDict provides static type checking without enforcing types at runtime
new_person: Person = {"namw": "Aryan", "age": 20}

# Print the dictionary contents
print(new_person)