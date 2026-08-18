# Script to create and serialize a reusable research paper summarization PromptTemplate to JSON
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class

# Define the PromptTemplate with dynamic input variables and custom formatting rules
template = PromptTemplate(
    template=""" Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length""",

    input_variables=["paper_input", "style_input", "length_input"],  # Specify the required placeholder variables
    validate_template=True  # Validate template string against input_variables
)

# Save the configured prompt template object to a local JSON file ('template.json')
template.save('template.json')