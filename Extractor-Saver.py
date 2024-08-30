from langchain_community.document_loaders import UnstructuredURLLoader
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

urls = ["https://brainlox.com/courses/category/technical"]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

structured_data = []
for doc in data:
    content = doc.page_content
    courses = re.findall(r'\$[0-9]+ per session\n(.*?)\n\n(.*?)\n(\d+ Lessons)', content, re.DOTALL)
    for course in courses:
        course_info = {
            "title": course[0].strip(),
            "description": course[1].strip(),
            "lessons": course[2].strip()
        }
        structured_data.append(course_info)

# verification
# for course in structured_data:
#     print(course)

#Creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = [course['description'] for course in structured_data]
embeddings = model.encode(descriptions)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))
faiss.write_index(index, "vector_store.index")

# for varificatin
print(f"Number of embeddings in the index: {index.ntotal}")

with open('structured_data.json', 'w') as f:
    json.dump(structured_data, f)
