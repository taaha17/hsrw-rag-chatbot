"""
Zero - AI Academic Advisor for Rhine-Waal University
Gradio Web Interface

This is the main web application that students interact with.
Zero provides personalized academic guidance for ISE students.

Features:
- Degree program & semester selection
- ChatGPT-style conversation interface
- Schedule queries with room codes and block dates
- Module information with full type names (Lecture, Exercise, etc.)
- Dark theme, minimalist design

To run: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
import json
import os
from typing import List, Tuple, Optional

# Import core chatbot logic
from chat import generate_chat_response, format_module_details_from_rag
from logic_engine import (
    detect_query_intent,
    find_code_by_name,
    extract_semester_criteria,
    get_modules_from_map,
    get_schedule_for_module,
    get_schedule_for_day
)
from config import (
    MODULE_MAP_PATH,
    CLASS_SCHEDULE_PATH,
    DEGREE_PROGRAMS,
    decode_class_type,
    format_room_info,
    chat_log,
    debug_log
)
from utils import HttpOllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from config import DB_PATH, EMBEDDING_MODEL

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Load module map and schedule data once at startup
try:
    with open(MODULE_MAP_PATH, "r", encoding="utf-8") as f:
        module_map = json.load(f)
    with open(CLASS_SCHEDULE_PATH, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)
    print(f"✅ Loaded {len(module_map)} modules and {len(schedule_data)} schedule entries")
    debug_log.info(f"Data loaded: {len(module_map)} modules, {len(schedule_data)} schedules")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    debug_log.error(f"Data loading failed: {e}")
    print("Please run 'python ingest.py' first to generate the data files.")
    exit(1)

# Initialize RAG retrievers
embeddings = HttpOllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

all_docs = vectorstore.get()
doc_objects = [Document(page_content=t, metadata=m) for t, m in zip(all_docs['documents'], all_docs['metadatas'])]

bm25_retriever = BM25Retriever.from_documents(doc_objects)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

print("✅ RAG retrievers initialized")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_message(content):
    """
    Extract plain text from Gradio 6.x message content.
    
    Gradio 6.x Chatbot uses multimodal format:
    - String: "Hello" → return as-is
    - List: [{'text': 'Hello', 'type': 'text'}] → extract text
    - Dict: {'text': 'Hello', 'type': 'text'} → extract text
    
    Args:
        content: Message content (str, list, or dict)
    
    Returns:
        Plain text string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from list of content parts
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        return ' '.join(text_parts)
    elif isinstance(content, dict):
        # Single content part
        return content.get('text', str(content))
    else:
        return str(content)

def format_schedule_entry(entry: dict) -> str:
    """
    Format a schedule entry for display with full type names and room details.
    
    Example output:
    "📅 Monday, 14:00-15:30
     📚 Signals and Systems (Lecture)
     👨‍🏫 Prof. Dr. Strumpen
     📍 Hörsaal, Building 1, Ground Floor, Room 215
     📆 Block dates: 29.09.25, 03.11.25, 24.11.25"
    """
    # Decode class type (L -> Lecture, E -> Exercise, etc.)
    type_full = decode_class_type(entry['type'])
    
    # Format room with building/floor/room codes
    room_info = format_room_info(
        entry['room'],
        entry.get('building'),
        entry.get('floor'),
        entry.get('room_number')
    )
    
    # Build formatted string
    formatted = f"""📅 {entry['day']}, {entry['start_time']}-{entry['end_time']}
📚 {entry['module_name']} ({type_full})
👨‍🏫 {entry['professor']}
📍 {room_info}"""
    
    # Add block dates if present
    if 'block_dates' in entry:
        formatted += f"\n📆 Block dates: {entry['block_dates']}"
    
    return formatted


def format_schedule_list(schedule_entries: List[dict]) -> str:
    """Format multiple schedule entries as a numbered list."""
    if not schedule_entries:
        return "No classes found."
    
    formatted_list = []
    for i, entry in enumerate(schedule_entries, 1):
        formatted_list.append(f"**Class {i}:**\n{format_schedule_entry(entry)}\n")
    
    return "\n".join(formatted_list)

# ═══════════════════════════════════════════════════════════════════════════
# CHAT LOGIC WITH USER CONTEXT
# ═══════════════════════════════════════════════════════════════════════════

def chat_with_zero(
    message: str,
    history: List[dict],
    degree_program: str,
    semester: int
) -> Tuple[List[dict], str]:
    """
    Main chat function that processes user messages with context.
    
    Args:
        message: User's question
        history: Conversation history as list of message dicts with 'role' and 'content'
        degree_program: Selected degree program code (e.g., 'ISE')
        semester: Selected semester number (1-7)
    
    Returns:
        Updated history and empty string for textbox
    """
    try:
        if not message.strip():
            return history, ""
        
        # === CRITICAL DEBUG: Log incoming history structure ===
        debug_log.info(f"="*60)
        debug_log.info(f"chat_with_zero called")
        debug_log.info(f"Message: {message[:80]}...")
        debug_log.info(f"History type: {type(history)}, length: {len(history)}")
        for i, item in enumerate(history):
            debug_log.info(f"  History[{i}]: type={type(item)}, keys={item.keys() if isinstance(item, dict) else 'N/A'}")
            if isinstance(item, dict):
                content = item.get('content', '<missing>')
                content_type = type(content).__name__
                debug_log.info(f"    role={item.get('role')}, content_type={content_type}, content_preview={str(content)[:50]}...")
        debug_log.info(f"="*60)
        
        # Log the incoming query
        chat_log.info(f"[UI] {message[:80]}... | S{semester}" if len(message) > 80 else f"[UI] {message} | S{semester}")
        
        # Build context based on query type (similar to chat.py logic)
        intent_analysis = detect_query_intent(message)
        intent = intent_analysis['intent']
        
        context = ""
        hardcoded_list = None
        
        # === SCHEDULE QUERIES ===
        if intent == 'schedule':
            module_code = find_code_by_name(message, module_map)
            
            if module_code:
                # Specific module schedule
                module_name = module_map.get(module_code, "")
                schedule_info = get_schedule_for_module(module_name, schedule_data, module_code)
                
                if schedule_info:
                    formatted_schedule = format_schedule_list(schedule_info)
                    context = f"""[OFFICIAL CLASS SCHEDULE]
Module: {module_name}

{formatted_schedule}

IMPORTANT: Present this information clearly to the student."""
                else:
                    context = f"[SYSTEM]: Module '{module_name}' exists but no schedule data available."
            
            else:
                # Day/semester based schedule
                import datetime
                import pytz
                
                # Get current day in Germany timezone
                germany_tz = pytz.timezone('Europe/Berlin')
                now = datetime.datetime.now(germany_tz)
                
                day_mapping = {
                    'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
                    'thursday': 'Thursday', 'friday': 'Friday',
                    'montag': 'Monday', 'dienstag': 'Tuesday', 'mittwoch': 'Wednesday',
                    'donnerstag': 'Thursday', 'freitag': 'Friday'
                }
                
                query_day = None
                for day_name, day_proper in day_mapping.items():
                    if day_name in message.lower():
                        query_day = day_proper
                        break
                
                # If "today" or no day specified, use current day
                if not query_day or 'today' in message.lower():
                    query_day = now.strftime('%A')
                    debug_log.info(f"Using current day: {query_day}")
                
                # Use user's semester if not specified in query
                filters = extract_semester_criteria(message)
                query_semester = filters["semester_num"] if filters["semester_num"] else semester
                
                schedule_info = get_schedule_for_day(query_semester, query_day, schedule_data)
                
                if schedule_info:
                    # schedule_info is a tuple (list, bool, str), extract the list
                    schedule_list = schedule_info[0] if isinstance(schedule_info, tuple) else schedule_info
                    formatted_schedule = format_schedule_list(schedule_list)
                    context = f"""[OFFICIAL CLASS SCHEDULE FOR {query_day.upper()}]
Semester {query_semester}

{formatted_schedule}

IMPORTANT: Present this schedule clearly to the student."""
                else:
                    context = f"[SYSTEM]: No classes scheduled for semester {query_semester} on {query_day}."
        
        # === MODULE LIST QUERIES ===
        elif intent == 'modules_list':
            filters = extract_semester_criteria(message)
            # Use user's semester if not specified
            if not filters["semester_num"]:
                filters["semester_num"] = semester
            
            if filters["semester_num"] or filters["season"]:
                hardcoded_list = get_modules_from_map(module_map, filters)
        
        # === MODULE INFO QUERIES ===
        elif intent == 'module_info':
            module_code = find_code_by_name(message, module_map)
            
            if module_code:
                module_name = module_map.get(module_code, "")
                
                # Retrieve detailed module info from RAG
                retrieved_docs = ensemble_retriever.invoke(message)
                
                # Use structured formatter for module details
                context = format_module_details_from_rag(module_code, module_name, retrieved_docs)
                
                # Also check if they want schedule info for this module
                if any(word in message.lower() for word in ['when', 'schedule', 'time', 'day']):
                    schedule_info = get_schedule_for_module(module_name, schedule_data, module_code)
                    
                    if schedule_info:
                        formatted_schedule = format_schedule_list(schedule_info)
                        context += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS SCHEDULE FOR THIS MODULE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{formatted_schedule}

IMPORTANT: Include this schedule information in your response."""
            else:
                # Module not found, use general RAG
                retrieved_docs = ensemble_retriever.invoke(message)
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # === GENERAL QUERIES ===
        else:
            retrieved_docs = ensemble_retriever.invoke(message)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
        # Add user context to the conversation
        user_context = f"\n\n[USER PROFILE]\nDegree: {degree_program}\nCurrent Semester: {semester}"
        context += user_context
        
        # Convert Gradio history format to chat.py format
        # Safely handle history - ensure all messages have correct structure
        # Gradio 6.x uses multimodal format: content can be list of dicts
        chat_history = []
        for msg in history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Extract plain text from Gradio's multimodal format
                plain_text = extract_text_from_message(msg['content'])
                chat_history.append({
                    'role': msg['role'],
                    'content': plain_text
                })
        
        # Generate response
        response = generate_chat_response(context, chat_history, message, hardcoded_list)
        
        # Update history with Gradio 6.0 format (dictionaries with role and content)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # === CRITICAL DEBUG: Verify history structure before return ===
        debug_log.info(f"Returning from chat_with_zero:")
        debug_log.info(f"  History length: {len(history)}")
        if len(history) > 0:
            last_item = history[-1]
            debug_log.info(f"  Last item type: {type(last_item)}")
            if isinstance(last_item, dict):
                debug_log.info(f"    role={last_item.get('role')}, content_type={type(last_item.get('content')).__name__}")
        
        return history, ""
    
    except Exception as e:
        # Log the error with full traceback
        debug_log.error(f"Error in chat_with_zero: {str(e)}", exc_info=True)
        chat_log.error(f"Failed to process message: {message}")
        
        # Return error message to user
        error_msg = "❌ Sorry, I encountered an error processing your request. Please try again or rephrase your question."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        
        return history, ""# ═══════════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def generate_follow_up_suggestions(last_query, last_response, intent):
    """
    Generate contextually relevant follow-up question suggestions.
    
    Uses simple rule-based logic to suggest relevant next questions based on
    what the user just asked.
    
    Args:
        last_query: The user's last question
        last_response: The bot's last response
        intent: The detected intent of the last query
    
    Returns:
        List of 3 suggested follow-up questions
    """
    
    # Default suggestions based on intent
    suggestions = {
        'schedule': [
            "What room is this class in?",
            "Who is the professor?",
            "Are there any block dates?"
        ],
        'module_info': [
            "When is this module scheduled?",
            "What are the prerequisites?",
            "How many credits is this worth?"
        ],
        'modules_list': [
            "Tell me more about [module name]",
            "What classes do I have today?",
            "Which modules are electives?"
        ],
        'general': [
            "What modules do I have this semester?",
            "When is my next class?",
            "Tell me about internship options"
        ]
    }
    
    # Get base suggestions for this intent
    base_suggestions = suggestions.get(intent, suggestions['general'])
    
    # Add context-specific suggestions
    contextual_suggestions = []
    
    # If they asked about a specific module, suggest related questions
    if 'machine learning' in last_query.lower():
        contextual_suggestions.append("What are the prerequisites for Machine Learning?")
        contextual_suggestions.append("When is Machine Learning taught?")
    elif 'physics' in last_query.lower():
        contextual_suggestions.append("What topics are covered in Physics?")
        contextual_suggestions.append("Who teaches Physics?")
    elif 'internship' in last_query.lower():
        contextual_suggestions.append("Can I do a semester abroad instead?")
        contextual_suggestions.append("How do I find an internship?")
    elif 'today' in last_query.lower() or 'tomorrow' in last_query.lower():
        contextual_suggestions.append("What's my full week schedule?")
        contextual_suggestions.append("Show me next week's classes")
    
    # Combine contextual and base suggestions
    if contextual_suggestions:
        final_suggestions = contextual_suggestions[:2] + [base_suggestions[0]]
    else:
        final_suggestions = base_suggestions
    
    return final_suggestions[:3]  # Return exactly 3 suggestions


def create_interface():
    """Create and configure the Gradio interface with custom theme and UI improvements."""
    
    with gr.Blocks(
        title="Zero - AI Academic Advisor"  # Browser tab title
    ) as demo:
        
        # === HEADER ===
        gr.Markdown(
            """
            # 🤖 Zero - AI Academic Advisor
            ### Rhine-Waal University of Applied Sciences
            
            Welcome! I'm Zero, your AI academic advisor. I can help you with:
            - 📅 Class schedules (times, rooms, professors, block dates)
            - 📚 Module information (content, prerequisites, credits)
            - 📝 General academic questions
            """,
            elem_id="welcome-box"
        )
        
        # === USER PROFILE SELECTION ===
        with gr.Row():
            degree_dropdown = gr.Dropdown(
                choices=[prog['name'] for prog in DEGREE_PROGRAMS.values()],
                value="Infotronic Systems Engineering",
                label="🎓 Degree Program",
                interactive=True
            )
            
            semester_dropdown = gr.Dropdown(
                choices=list(range(1, 8)),
                value=1,
                label="📖 Current Semester",
                interactive=True
            )
        
        # === CHAT INTERFACE ===
        chatbot = gr.Chatbot(
            value=[],
            elem_id="chatbot",
            show_label=False,
            height=500
        )
        
        # === SUGGESTED FOLLOW-UP PROMPTS ===
        # Display as clickable buttons
        with gr.Row(visible=False, elem_id="suggestions-row") as suggestions_row:
            suggestion_btn1 = gr.Button("", size="sm", visible=False, elem_id="sugg1")
            suggestion_btn2 = gr.Button("", size="sm", visible=False, elem_id="sugg2")
            suggestion_btn3 = gr.Button("", size="sm", visible=False, elem_id="sugg3")
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me anything about your studies... (e.g., 'What classes do I have today?')",
                show_label=False,
                scale=9,
                container=False
            )
            submit = gr.Button("Send", scale=1, variant="primary")
        
        # === EXAMPLE QUERIES ===
        gr.Examples(
            examples=[
                "What modules do I have in my current semester?",
                "When is my Signals and Systems class?",
                "What classes do I have today?",
                "Tell me about the Data Science module",
                "Who teaches Machine Learning?",
                "Which semester is the internship in?"
            ],
            inputs=msg,
            label="💡 Example Questions"
        )
        
        # === EVENT HANDLERS ===
        def on_submit(message, history, degree, semester):
            # Convert degree name to code
            degree_code = None
            for code, prog in DEGREE_PROGRAMS.items():
                if prog['name'] == degree:
                    degree_code = code
                    break
            
            # Get response from chatbot
            new_history, empty_textbox = chat_with_zero(message, history, degree_code or 'ISE', semester)
            
            # Generate follow-up suggestions
            if len(new_history) >= 2:
                last_user_content = new_history[-2].get('content', '')
                last_bot_content = new_history[-1].get('content', '')
                
                # Extract plain text from Gradio 6.x multimodal format
                last_user_msg = extract_text_from_message(last_user_content)
                last_bot_msg = extract_text_from_message(last_bot_content)
                
                debug_log.debug(f"Generating suggestions for: {last_user_msg[:50]}...")
                
                # Detect intent for suggestions
                intent_analysis = detect_query_intent(last_user_msg)
                intent = intent_analysis['intent']
                
                # Generate suggestions
                suggestions = generate_follow_up_suggestions(last_user_msg, last_bot_msg, intent)
                
                # Return with visible buttons containing suggestion text
                return (
                    new_history, 
                    empty_textbox, 
                    gr.update(visible=True),  # suggestions_row
                    gr.update(value=suggestions[0], visible=True),  # btn1
                    gr.update(value=suggestions[1] if len(suggestions) > 1 else "", visible=len(suggestions) > 1),  # btn2
                    gr.update(value=suggestions[2] if len(suggestions) > 2 else "", visible=len(suggestions) > 2)   # btn3
                )
            
            # No suggestions - hide all buttons
            return (
                new_history, 
                empty_textbox, 
                gr.update(visible=False),  # suggestions_row
                gr.update(visible=False),  # btn1
                gr.update(visible=False),  # btn2
                gr.update(visible=False)   # btn3
            )
        
        # Helper function to handle suggestion button clicks
        def on_suggestion_click(suggestion_text, history, degree, semester):
            # Same as on_submit but with the suggestion text as the message
            return on_submit(suggestion_text, history, degree, semester)
        
        # Wire up submit button and enter key
        submit.click(
            on_submit,
            inputs=[msg, chatbot, degree_dropdown, semester_dropdown],
            outputs=[chatbot, msg, suggestions_row, suggestion_btn1, suggestion_btn2, suggestion_btn3]
        )
        
        msg.submit(
            on_submit,
            inputs=[msg, chatbot, degree_dropdown, semester_dropdown],
            outputs=[chatbot, msg, suggestions_row, suggestion_btn1, suggestion_btn2, suggestion_btn3]
        )
        
        # Wire up suggestion buttons to send their text as a message
        suggestion_btn1.click(
            on_suggestion_click,
            inputs=[suggestion_btn1, chatbot, degree_dropdown, semester_dropdown],
            outputs=[chatbot, msg, suggestions_row, suggestion_btn1, suggestion_btn2, suggestion_btn3]
        )
        
        suggestion_btn2.click(
            on_suggestion_click,
            inputs=[suggestion_btn2, chatbot, degree_dropdown, semester_dropdown],
            outputs=[chatbot, msg, suggestions_row, suggestion_btn1, suggestion_btn2, suggestion_btn3]
        )
        
        suggestion_btn3.click(
            on_suggestion_click,
            inputs=[suggestion_btn3, chatbot, degree_dropdown, semester_dropdown],
            outputs=[chatbot, msg, suggestions_row, suggestion_btn1, suggestion_btn2, suggestion_btn3]
        )
    
    return demo

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("🚀 Starting Zero - AI Academic Advisor")
    print("="*80)
    
    debug_log.info("Zero web application started")
    
    demo = create_interface()
    
    # Launch the app (Gradio 6.0 compatible)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css="""
        /* Hide Gradio footer buttons (API, Built with Gradio, Settings) */
        footer {
            display: none !important;
        }
        .gradio-container footer {
            display: none !important;
        }
        /* Hide the "Use via API" and other footer buttons */
        .footer {
            display: none !important;
        }
        .contain {
            max-width: 1200px !important;
        }
        /* Style the welcome box */
        #welcome-box {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-bottom: 20px;
        }
        /* Style suggestion buttons like modern chat UI */
        #suggestions-row {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        #suggestions-row button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 8px 16px !important;
            color: white !important;
            font-size: 14px !important;
            transition: all 0.3s ease !important;
            margin: 5px !important;
        }
        #suggestions-row button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        }
        """
    )
