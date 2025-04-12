import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)

def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_content(content: str) -> List[str]:
    """Split content by '---' separator."""
    # Using regex to handle different styles of separators with whitespace
    regex = r'\n\s*---\s*\n'
    sections = re.split(regex, content)
    return [section.strip() for section in sections if section.strip()]

def process_section(section: str, index: int) -> Dict[str, Any]:
    """Process a section to extract title and category."""
    lines = section.split('\n')
    title = lines[0].strip() if lines else "Unknown Title"
    
    # Determine category based on the content
    category = "Other"
    if "experience" in title.lower() and "teaching" not in title.lower():
        category = "Work Experience"
    elif "project" in title.lower():
        category = "Projects"
    elif "education" in title.lower():
        category = "Education"
    elif "teaching" in title.lower():
        category = "Teaching"
    elif "leadership" in title.lower():
        category = "Leadership"
    elif "skills" in title.lower():
        category = "Skills"
    elif "profile" in title.lower():
        category = "Personal Information"
    
    return {
        "id": f"section-{index + 1}",
        "title": title,
        "content": section,
        "category": category,
        "metadata": {
            "source": "resume",
            "section_number": index + 1
        }
    }

def sections_to_documents(sections: List[Dict[str, Any]]) -> List[Document]:
    """Convert sections to LangChain Documents."""
    documents = []
    for section in sections:
        doc = Document(
            page_content=section["content"],
            metadata={
                "id": section["id"],
                "title": section["title"],
                "category": section["category"],
                "source": section["metadata"]["source"],
                "section_number": section["metadata"]["section_number"]
            }
        )
        documents.append(doc)
    return documents

def save_sections_to_json(sections: List[Dict[str, Any]], output_file: str):
    """Save processed sections to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2)
    print(f"Saved {len(sections)} sections to {output_file}")

def upload_to_pinecone(sections: List[Dict[str, Any]]):
    """Upload sections to Pinecone using the updated API format."""
    # Initialize Pinecone with the new format
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    try:
        # Try to connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Connected to existing index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Error connecting to index: {e}")
        print(f"Creating new index: {PINECONE_INDEX_NAME}")
        # Create the index if it doesn't exist
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Dimension for Gemini embeddings
            metric="cosine"
        )
        index = pc.Index(PINECONE_INDEX_NAME)
    
    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Convert sections to documents
    documents = sections_to_documents(sections)
    
    # Process documents and create vectors for upsert
    vectors = []
    batch_size = 100  # Process in batches to avoid memory issues
    
    print(f"Processing {len(documents)} documents...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        for j, doc in enumerate(batch):
            # Generate embedding for document content
            embedding = embeddings.embed_query(doc.page_content)
            
            # Create vector record
            vector = {
                "id": doc.metadata["id"],
                "values": embedding,
                "metadata": {
                    "title": doc.metadata["title"],
                    "category": doc.metadata["category"],
                    "content": doc.page_content,
                    "source": doc.metadata["source"],
                    "section_number": doc.metadata["section_number"]
                }
            }
            vectors.append(vector)
        
        # Upsert batch of vectors
        print(f"Upserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        index.upsert(vectors=vectors)
        vectors = []  # Clear for next batch
    
    print(f"Successfully uploaded {len(documents)} sections to Pinecone index '{PINECONE_INDEX_NAME}'.")

def main():
    # File path
    input_file = "RAG_Splits_LoZhiHao.txt"  # Change this to your file path
    output_file = "resume_sections.json"
    
    # Step 1: Read the file
    print(f"Reading file: {input_file}")
    content = read_file(input_file)
    
    # Step 2: Split the content into sections
    print("Splitting content into sections...")
    raw_sections = split_content(content)
    print(f"Found {len(raw_sections)} sections")
    
    # Step 3: Process each section
    print("Processing sections...")
    processed_sections = [process_section(section, i) for i, section in enumerate(raw_sections)]
    
    # Step 4: Save to JSON for reference
    save_sections_to_json(processed_sections, output_file)
    
    # Step 5: Upload to Pinecone
    print("Uploading to Pinecone...")
    upload_to_pinecone(processed_sections)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()