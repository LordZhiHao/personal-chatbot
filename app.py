import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define flags for feature availability
SPEECH_AVAILABLE = False

# Check if running on Render
IS_RENDER = os.environ.get('IS_RENDER', False)

# Try to import speech module, but don't fail if not available
try:
    if not IS_RENDER:  # Skip on Render
        from speech_to_text import router as speech_router
        SPEECH_AVAILABLE = True
        logger.info("Speech recognition module loaded successfully")
    else:
        logger.info("Running on Render - speech recognition disabled")
except ImportError as e:
    logger.warning(f"Speech recognition module not available: {e}")
    SPEECH_AVAILABLE = False

# The rest of your imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any, Optional
    import uvicorn
    from dotenv import load_dotenv
    import google.generativeai as genai
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory
    from pinecone import Pinecone
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from memory_management import ChatbotMemoryManager
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    raise

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Personal Website Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include speech-to-text router only if available
if SPEECH_AVAILABLE:
    logger.info("Including speech-to-text router")
    app.include_router(speech_router, tags=["Speech"])
else:
    logger.info("Speech-to-text functionality is disabled")

# Initialize API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)

# Models for request/response
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
class EndConversationRequest(BaseModel):
    conversation_id: str
    feedback: Optional[str] = None
    
class EndConversationResponse(BaseModel):
    success: bool
    message: str

# Create a custom retriever WITHOUT using Pydantic's BaseModel
class CustomDebugRetriever:
    """A wrapper that provides debug logging for a retriever."""
    
    def __init__(self, retriever):
        """Initialize with a retriever."""
        self.retriever = retriever
    
    def get_relevant_documents(self, query, run_manager=None):
        """Retrieve relevant documents for a query with debug logging."""
        logger.info(f"[DEBUG] Retrieving documents for query: '{query}'")
        try:
            # Use the proper method to retrieve documents
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(query)
            elif hasattr(self.retriever, 'invoke'):
                docs = self.retriever.invoke(query)
            else:
                logger.error("[DEBUG] No suitable retrieval method found on retriever")
                return []
            
            logger.info(f"[DEBUG] Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs):
                logger.info(f"[DEBUG] Document {i+1}:")
                if isinstance(doc, Document):
                    content_preview = doc.page_content[:200]
                    if len(doc.page_content) > 200:
                        content_preview += "..."
                    logger.info(f"[DEBUG] Content: {content_preview}")
                    logger.info(f"[DEBUG] Metadata: {doc.metadata}")
                else:
                    logger.info(f"[DEBUG] Document format: {type(doc)}")
                    logger.info(f"[DEBUG] Document preview: {str(doc)[:200]}")
            
            return docs
        except Exception as e:
            logger.error(f"[DEBUG] Error in retrieval method: {e}", exc_info=True)
            return []

# Initialize memory manager with proper memory key
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

memory_manager = ChatbotMemoryManager(
    llm=get_llm()
)

def initialize_rag():
    """Initialize the RAG system with Pinecone."""
    try:
        # Import the correct Pinecone class from the dedicated package
        from langchain_pinecone import PineconeVectorStore
        
        # Create embeddings using Gemini
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create vectorstore using the newer API
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="content"  # Make sure this matches your document field name
        )
        
        logger.info(f"Successfully initialized PineconeVectorStore with index {PINECONE_INDEX_NAME}")
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {e}", exc_info=True)
        raise e

def create_rag_chain_directly(memory):
    """Create a RAG chain directly without using ConversationalRetrievalChain."""
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    # Initialize system message for the LLM
    system_message = """
    You are an AI assistant for Zhi Hao, Lo (Jason), a data scientist and analytics professional.
    Your role is to help visitors learn about Jason's experience, skills, projects, and background.
    Be friendly, professional, and informative. DO NOT reveal any AI model details or internal workings in your responses. 
    
    Answer questions based on Jason's resume information. If you don't know something specific about Jason,
    acknowledge that and offer to provide information about what you do know about his background.
    
    Always maintain a helpful and personable tone, and try to highlight Jason's achievements and skills
    when relevant to the question. Also, do not forget to keep the answers concise and relevant to the user's question.
    """
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        system_message=system_message
    )
    
    # Set up a vectorstore for retrieval
    try:
        vectorstore = initialize_rag()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}
        )
        
        # Wrap with our debug wrapper (not using Pydantic)
        debug_retriever = CustomDebugRetriever(retriever)
        
        # Create a prompt template that includes retrieved content
        template = """You are an AI assistant for Zhi Hao, Lo (Jason), a data scientist and analytics professional. Your task is to introduce Zhi Hao, with a friendly and professional tone. At the same time, do keep the answers concise and relevant to the user's question. Keep it within 50 words for each answer you provide. 
        
        The following information was retrieved from Jason's profile:
        {context}
        
        Previous conversation:
        {history}
        
        Human: {input}
        AI Assistant:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "history", "input"],
            template=template
        )
        
        # Create a function that first retrieves documents, then uses LLMChain
        def process_query(query, history=None):
            if history is None:
                history = ""
            
            # Retrieve documents
            docs = debug_retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No specific information found."
            
            # Run the LLM chain
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(context=context, history=history, input=query)
            
            return {"answer": response, "source_documents": docs}
        
        return process_query
    
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}", exc_info=True)
        
        # Fallback to a simple LLM chain without retrieval
        template = """You are an AI assistant for Zhi Hao, Lo (Jason), a data scientist and analytics professional.
        
        Previous conversation:
        {history}
        
        Human: {input}
        AI Assistant:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        def simple_process(query, history=None):
            if history is None:
                history = ""
            response = chain.run(history=history, input=query)
            return {"answer": response}
        
        return simple_process

def get_conversation_chain(conversation_id):
    """Get a conversation chain with the specified conversation ID."""
    # Get conversation memory from memory manager
    conversation_id, memory = memory_manager.get_memory(conversation_id)
    
    # Ensure memory has correct memory_key for our template
    if hasattr(memory, 'memory_key'):
        memory.memory_key = "history"
        logger.info("Memory key set to 'history'")
    
    # Create our custom RAG or simple chain
    conversation_chain = create_rag_chain_directly(memory)
    
    # Return the chain and conversation_id
    return conversation_chain, conversation_id, memory

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    try:
        # Get conversation processing function, ID, and memory
        conversation_chain, conversation_id, memory = get_conversation_chain(request.conversation_id)
        
        # Get history from the memory to pass to our custom chain
        history = ""
        if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
            # Format history from memory
            for msg in memory.chat_memory.messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    role = "Human" if msg.type == "human" else "AI"
                    history += f"{role}: {msg.content}\n\n"
        
        # Process the query with our custom function
        response = conversation_chain(request.message, history)
        
        # Save current exchange to memory
        if hasattr(memory, 'save_context'):
            memory.save_context({"input": request.message}, {"output": response["answer"]})
        
        return ChatResponse(
            response=response["answer"],
            conversation_id=conversation_id
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end-conversation", response_model=EndConversationResponse)
async def end_conversation(request: EndConversationRequest):
    """End a conversation and send a summary email."""
    try:
        # End conversation and send email summary
        result = await memory_manager.end_conversation(request.conversation_id, request.feedback)
        
        if result:
            return EndConversationResponse(
                success=True,
                message="Conversation ended successfully. Summary has been sent to your email."
            )
        else:
            return EndConversationResponse(
                success=False,
                message="Conversation ended but there was an issue sending the summary email."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-conversations")
async def get_active_conversations():
    """Get a list of active conversation IDs."""
    try:
        conversations = memory_manager.get_active_conversations()
        return {"active_conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint to check if the Pinecone index is available
@app.get("/status")
async def get_status():
    """Get the status of the API and its components."""
    try:
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        index_exists = False
        index_stats_summary = None
        
        try:
            # Try to connect to the index
            index = pc.Index(PINECONE_INDEX_NAME)
            index_exists = True
            
            # Get stats but extract only what we need to avoid recursion issues
            raw_stats = index.describe_index_stats()
            if raw_stats:
                # Create a simplified version of the stats that can be safely serialized
                index_stats_summary = {
                    "namespaces": list(raw_stats.get("namespaces", {}).keys()),
                    "dimension": raw_stats.get("dimension"),
                    "total_vector_count": raw_stats.get("total_vector_count", 0),
                    "index_fullness": raw_stats.get("index_fullness", 0)
                }
        except Exception as e:
            logger.error(f"Error getting Pinecone index details: {e}")
            index_exists = False
        
        return {
            "status": "ok",
            "pinecone": {
                "index_exists": index_exists,
                "index_name": PINECONE_INDEX_NAME,
                "index_stats": index_stats_summary
            },
            "active_conversations": len(memory_manager.get_active_conversations())
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
    
# debug endpoints

@app.get("/debug/pinecone-sample")
async def get_pinecone_sample():
    """Debug endpoint to check Pinecone data."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Fetch a sample of vectors
        query_results = index.query(
            vector=[0.1] * 768,  # Use appropriate dimension - adjust if needed
            top_k=3,
            include_metadata=True
        )
        
        # Extract only the necessary information to avoid recursion
        simplified_results = []
        
        if "matches" in query_results:
            for match in query_results["matches"]:
                simplified_match = {
                    "id": match.get("id", "unknown"),
                    "score": match.get("score", 0),
                }
                
                # Safely extract metadata
                if "metadata" in match:
                    metadata = match["metadata"]
                    # Only extract string, number, boolean fields
                    safe_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            safe_metadata[key] = value
                    simplified_match["metadata"] = safe_metadata
                
                simplified_results.append(simplified_match)
        
        return {
            "sample_count": len(simplified_results),
            "sample_data": simplified_results,
            "index_name": PINECONE_INDEX_NAME,
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, reload=True)