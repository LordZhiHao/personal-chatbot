# memory_management.py
# We'll keep using ConversationBufferMemory even with the deprecation warning
# Since migrating to RunnableWithMessageHistory would require more extensive changes
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "your-email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "your-app-password")  # App password for Gmail
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "your-email@gmail.com")

class ChatbotMemoryManager:
    def __init__(self, llm):
        """
        Initialize the memory manager for the chatbot.
        
        Args:
            llm: The language model to use for summarization
        """
        self.llm = llm
        self.active_memories = {}
    
    def get_memory(self, conversation_id: Optional[str] = None) -> tuple:
        """
        Get or create a memory instance for a conversation.
        
        Args:
            conversation_id (str, optional): Unique identifier for the conversation
            
        Returns:
            tuple: (conversation_id, memory_instance)
        """
        # Generate a new conversation ID if none provided
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # Check if memory already exists for this conversation
        if conversation_id in self.active_memories:
            return conversation_id, self.active_memories[conversation_id]
        
        # Create new memory
        # Note: There's a deprecation warning here but we'll keep using it for now
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Store in active memories
        self.active_memories[conversation_id] = memory
        return conversation_id, memory
    
    async def end_conversation(self, conversation_id: str, user_feedback: Optional[str] = None) -> bool:
        """
        End a conversation, summarize it, and send summary to email.
        
        Args:
            conversation_id (str): Unique identifier for the conversation
            user_feedback (str, optional): Optional feedback from the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        if conversation_id not in self.active_memories:
            return False
        
        memory = self.active_memories[conversation_id]
        
        # Get chat history
        messages = []
        if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
            messages = memory.chat_memory.messages
            
        # Format messages for summarization
        conversation_text = ""
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            conversation_text += f"{role}: {msg.content}\n\n"
        
        # Generate summary
        summary = await self._generate_summary(conversation_text)
        
        # Send email with summary
        success = self._send_email_summary(summary, conversation_text, user_feedback)
        
        # Remove from active memories
        if conversation_id in self.active_memories:
            del self.active_memories[conversation_id]
        
        return success
    
    async def _generate_summary(self, conversation_text: str) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            conversation_text (str): The full conversation text
            
        Returns:
            str: A summary of the conversation
        """
        # Use LLM to generate summary
        try:
            # Create a prompt for summarization
            prompt = f"""
            Please provide a concise summary of the following conversation. 
            Include the main topics discussed, any questions asked and answers provided, 
            and any action items or follow-ups needed.
            
            CONVERSATION:
            {conversation_text}
            
            SUMMARY:
            """
            
            # Use the LLM to generate the summary
            response = await self.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Failed to generate summary: {str(e)}\n\nRAW CONVERSATION:\n{conversation_text}"
    
    def _send_email_summary(self, summary: str, full_conversation: str, user_feedback: Optional[str] = None) -> bool:
        """
        Send an email with the conversation summary.
        
        Args:
            summary (str): Summary of the conversation
            full_conversation (str): The full conversation text
            user_feedback (str, optional): Optional feedback from the user
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_USER
            msg['To'] = EMAIL_RECIPIENT
            msg['Subject'] = f"Chatbot Conversation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Email body
            body = f"""
            <h2>Conversation Summary</h2>
            <p>{summary}</p>
            
            {"<h3>User Feedback</h3><p>" + user_feedback + "</p>" if user_feedback else ""}
            
            <h3>Full Conversation</h3>
            <pre style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            {full_conversation}
            </pre>
            
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to server and send email
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def get_active_conversations(self) -> List[str]:
        """
        Get list of active conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        return list(self.active_memories.keys())