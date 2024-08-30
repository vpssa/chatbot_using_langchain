from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Load the model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("vector_store.index")

# Load the structured data from a JSON file
with open('structured_data.json', 'r') as f:
    structured_data = json.load(f)

# Initialize OpenAI LLM
openai_llm = OpenAI(api_key='your_openai_api_key')

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["course_title", "course_description", "lessons"],
    template="Course Title: {course_title}\nDescription: {course_description}\nLessons: {lessons}\n\nCan you provide more details about this course in a conversational manner?"
)

# Create an LLMChain
llm_chain = LLMChain(llm=openai_llm, prompt=prompt_template)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_embedding = model.encode([user_input])

    # Search the vector store
    D, I = index.search(np.array(user_embedding), k=1)
    course_info = structured_data[I[0][0]]

    # Generate a human-like response
    response = llm_chain.run({
        "course_title": course_info["title"],
        "course_description": course_info["description"],
        "lessons": course_info["lessons"]
    })

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=1111)
