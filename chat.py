"""
Academic Advisor Chatbot for Infotronic Systems Engineering (ISE)

This is the main chatbot interface that combines:
1. Structured data (class schedules, module lists) for precise queries
2. RAG (Retrieval-Augmented Generation) for detailed module information
3. Smart query routing to prioritize the right data source

The chatbot can answer questions like:
- "When is my Signals and Systems class?"
- "What modules do I have in 2nd semester?"
- "Tell me about the Data Science module"
- "What classes do I have today?"

Key Design Principles:
- PRIORITIZE structured data over RAG for factual queries (schedules, lists)
- Use clear, directive prompts to prevent LLM hallucinations
- Detect user intent first, then route to appropriate data source
- Provide contextual help (e.g., ask for semester if not specified)
"""

import os
import json
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from logic_engine import (
    extract_semester_criteria, 
    get_modules_from_map, 
    find_code_by_name,
    get_schedule_for_module,
    get_schedule_for_day,
    get_all_schedule_for_semester,
    detect_query_intent
)
from utils import HttpOllamaEmbeddings
from config import (
    DB_PATH,
    MODULE_MAP_PATH,
    CLASS_SCHEDULE_PATH,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    OLLAMA_URL,
    chat_log,
    debug_log,
    prompt_log
)


def format_module_details_from_rag(module_code, module_name, rag_docs):
    """
    Format module information in the structured handbook format.
    
    Extracts and organizes information from RAG documents into the standard
    module handbook structure with 11 key fields.
    
    Args:
        module_code: Module code (e.g., 'CI_3.01')
        module_name: Full module name
        rag_docs: List of retrieved documents from RAG
    
    Returns:
        Formatted string with module details in handbook structure
    """
    
    # Combine all RAG docs
    combined_text = "\n\n".join([doc.page_content for doc in rag_docs])
    
    # Template for structured module information
    module_template = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE INFORMATION: {module_name} ({module_code})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 EXTRACTED FROM MODULE HANDBOOK:

{combined_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT INSTRUCTIONS FOR PRESENTING THIS MODULE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Present the information in this structure:

1. **Basic Information**
   - Code: {module_code}
   - Workload: [Extract from "Workload" field - total hours per semester]
   - Credits/ECTS: [Extract from "Credits" field - these are the SAME thing]
   - Level/Semester: [Extract from "Level of Module" field]
   - Frequency: [Extract from "Frequency of offer" - Winter/Summer semester]
   - Duration: [Usually 1 semester - courses run for ONE semester only unless explicitly stated]

2. **Course Structure**
   - Courses: [Extract from "Courses" - e.g., "4L + 2E" means 4 hours lecture + 2 hours exercise]
   - Teaching Time: [Official class hours per week]
   - Self-study: [Recommended self-study hours]
   - Group Size: [Maximum students in class]

3. **Learning Outcomes / Competences**
   [Extract from "Learning outcomes / Competences" section]

4. **Content**
   [Extract from "Content" section - topics taught]

5. **Teaching Methods**
   [Extract from "Teaching methods" - Lectures, Exercises, Seminars, etc.]

6. **Entry Requirements**
   [Extract from "Entry requirements" - prerequisites from previous semesters]
   ⚠️ Note: First semester courses typically have NO entry requirements

7. **Assessment**
   [Extract from "Types of assessment" - Graded/Certification/Oral exam, etc.]

8. **Requirements for Credit Points**
   [Extract from "Requirements for the award of credit points"]

9. **Professor in Charge**
   [Extract from "Person in charge of module"]

10. **Additional Information**
    [Extract from "Additional Information" - recommended books, papers, etc.]

⚠️ CRITICAL REMINDERS:
- Duration is ALWAYS 1 semester unless explicitly stated otherwise in the handbook
- Credit points and ECTS are the SAME thing (use the value from the handbook)
- Type of assessment must match EXACTLY what's in the handbook
- Entry requirements: Only list if explicitly stated, otherwise say "None" or "Not specified"
- Workload includes BOTH teaching time AND self-study hours
"""
    
    return module_template


def generate_chat_response(context, history, question, hardcoded_list=None):
    """
    Generates a chat response using the Ollama LLM with carefully structured prompts.
    
    The key to good responses is:
    1. Providing structured data clearly marked as authoritative
    2. Giving explicit instructions on how to use that data
    3. Not letting the LLM hallucinate when we have exact data
    """
    
    # 1. Dynamic List Injection
    list_instruction = ""
    if hardcoded_list:
        list_instruction = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OFFICIAL MODULE LIST - USE EXACTLY AS SHOWN]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(hardcoded_list)}

MANDATORY INSTRUCTIONS FOR THIS LIST:
1. Present this list EXACTLY as shown above
2. Do NOT add extra information about credit points, prerequisites, or other details unless the Context provides them
3. If the list is empty, tell the user no modules match their criteria
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """

    # 2. Global Knowledge (The "Common Sense" layer)
    import datetime
    import pytz
    
    # Get current date and time in Germany
    germany_tz = pytz.timezone('Europe/Berlin')
    now = datetime.datetime.now(germany_tz)
    current_date = now.strftime('%A, %B %d, %Y')
    current_time = now.strftime('%H:%M')
    current_day = now.strftime('%A')
    
    global_knowledge = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT DATE & TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Today is: {current_date}
Current time: {current_time} (Germany, Kamp-Lintfort)
Current day: {current_day}

When a student asks "What classes do I have today?", use {current_day} to look up their schedule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEGREE PROGRAM FACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Program: Infotronic Systems Engineering (ISE)
Institution: Rhine-Waal University of Applied Sciences, Kamp-Lintfort Campus
Duration: 7 Semesters (3.5 years)
Total Credits: 210 ECTS
Language: English

SEMESTER STRUCTURE:
- Semesters 1-3: Foundation courses (Mathematics, Physics, Programming, Electronics)
- Semesters 4-5: Advanced topics + Electives (you choose specialization areas)
- Semester 6: Internship OR Study Abroad (30 ECTS, student chooses ONE option)
- Semester 7: Bachelor thesis (12 ECTS) + Bachelor workshops/colloquium

MODULE CODE FORMAT:
- CI_X.YY where X = semester number
- CI_W.YY = Elective modules (typically taken in semesters 4-5)
- CI_K.YY = Key Competence modules (soft skills, languages, business)

SEMESTER SEASONS:
- Winter Semesters: 1, 3, 5, 7 (start in September/October)
- Summer Semesters: 2, 4, 6 (start in March/April)
- Students can take modules from earlier semesters if the season matches

CLASS TYPES:
- L = Lecture (Vorlesung)
- E = Exercise/Tutorial (Übung) 
- P = Practical/Lab (Praktikum)
- L&E = Combined Lecture and Exercise
- PT = Lab Project
- SL = Self-Learning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    # 3. Build the system prompt based on what data we have
    if "[OFFICIAL CLASS SCHEDULE" in context or "SCHEDULE_INFO" in context or "SEMESTER_MISMATCH" in context:
        # We have schedule data - emphasize using it correctly
        system_prompt = f"""you are zero, the academic advisor bot for rhine-waal university's infotronic systems engineering program.

{global_knowledge}

--- CORE RULES ---
1. BE CONCISE: answer directly, no unnecessary context
2. BE FACTUAL: only use the data provided below, never speculate
3. NO FLUFF: don't say "according to your profile" or "since it's thursday" unless directly relevant
4. when data shows no classes: state it clearly and helpfully
5. when semester doesn't match season: explain winter vs summer semesters

--- RESPONSE STYLE ---
GOOD: "your signals class is monday 14:00-15:30 (lecture) and 16:00-17:30 (exercise), hörsaal building 1, prof. dr. strumpen."
BAD: "according to your student profile, since you're in semester 2 and it's thursday which falls under summer semester season, let me check..."

GOOD: "you're in 2nd semester which runs in summer. winter is for semesters 1, 3, 5, 7. enjoy your break!"
BAD: "i apologize that i don't have information directly available. however, according to your student profile..."

{list_instruction}

--- DATA PROVIDED ---
{context}

student question: {question}
"""
    else:
        # Regular query - use standard prompt
        system_prompt = f"""you are zero, academic advisor bot for rhine-waal university's infotronic systems engineering.

{global_knowledge}
{list_instruction}

--- CORE RULES ---
1. BE CONCISE: answer directly, no fluff
2. BE FACTUAL: only use provided data, never speculate or add general knowledge
3. if asking about a specific module, use rhine-waal's data only, not general definitions
4. if data missing, say "i don't have that in the university documents"
5. when listing modules, present complete list without commentary

--- DATA AVAILABLE ---
{context}

student question: {question}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add ONLY user and assistant messages from history (skip old system prompts)
    for msg in history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append(msg)
    
    # Add the new user message
    messages.append({"role": "user", "content": question})
    
    # Log the conversation for debugging (concise)
    chat_log.info(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
    
    # Log the full prompt for refinement (only first 300 chars of system prompt)
    prompt_log.info("="*80)
    prompt_log.info(f"USER: {question}")
    prompt_log.info(f"SYSTEM PROMPT (first 300 chars):\n{system_prompt[:300]}...")
    if context:
        prompt_log.info(f"CONTEXT LENGTH: {len(context)} chars")
    
    data = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False
    }
    
    try:
        # Log message structure for debugging with type safety
        debug_log.debug(f"Sending {len(messages)} messages to Ollama API")
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            content_type = type(content).__name__
            # Safe string conversion
            if isinstance(content, str):
                content_preview = content[:100]
            else:
                content_preview = str(content)[:100]
            debug_log.debug(f"  Msg {i}: role={msg.get('role')}, type={content_type}, len={len(str(content))}")
        
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, timeout=60)
        response.raise_for_status()  # Raise exception for bad status codes
        
        bot_response = response.json()["message"]["content"]
        
        # Log the response (first 150 chars)
        chat_log.info(f"A: {bot_response[:150]}..." if len(bot_response) > 150 else f"A: {bot_response}")
        prompt_log.info(f"RESPONSE ({len(bot_response)} chars):\n{bot_response}")
        prompt_log.info("="*80)
        
        return bot_response
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Ollama API error: {str(e)}"
        debug_log.error(error_msg)
        
        # Try to get more details from the response
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                debug_log.error(f"Ollama error details: {error_detail}")
            except:
                debug_log.error(f"Ollama response text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
        
        chat_log.error(f"Failed to generate response for: {question}")
        return f"❌ Chat Error: Could not connect to Ollama. Please ensure Ollama is running."
    except KeyError as e:
        error_msg = f"Unexpected response format from Ollama: {e}"
        debug_log.error(error_msg)
        return f"❌ Chat Error: Unexpected response format from LLM"
    except Exception as e:
        error_msg = f"Unexpected error in generate_chat_response: {str(e)}"
        debug_log.error(error_msg, exc_info=True)
        return f"❌ Chat Error: {str(e)}"

def chat_loop():
    # Load module map and schedule data once
    try:
        with open(MODULE_MAP_PATH, "r") as f:
            module_map = json.load(f)
        with open(CLASS_SCHEDULE_PATH, "r") as f:
            schedule_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: module_map.json or class_schedule.json not found.")
        print("Please run the ingestion script first: python ingest.py")
        return

    # Initialize retrievers
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

    print(f"✅ Ready! (Indexed {len(module_map)} modules)")
    print("------------------------------------------------")

    history = []
    
    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break

            # --- SMART QUERY ROUTING ---
            # This is the brain of the chatbot - it decides what data to use based on the query
            hardcoded_list = None
            context = ""
            
            # Step 1: Detect what the user is asking for
            intent_analysis = detect_query_intent(query)
            intent = intent_analysis['intent']
            
            # Step 2: Route to the appropriate data source
            
            if intent == 'schedule':
                # USER WANTS SCHEDULE INFORMATION
                # Try to find a specific module first
                module_code = find_code_by_name(query, module_map)
                
                if module_code:
                    # User is asking about a specific module's schedule
                    module_name = module_map.get(module_code, "")
                    schedule_info = get_schedule_for_module(module_name, schedule_data, module_code)
                    
                    if schedule_info:
                        context = f"""[OFFICIAL CLASS SCHEDULE DATA]
Module: {module_name}
Schedule Details:
{json.dumps(schedule_info, indent=2)}

IMPORTANT: Use this data to answer the user's question. Tell them the exact day, time, professor, room, and type of class."""
                    else:
                        context = f"[SYSTEM]: Module '{module_name}' found in catalog but no schedule data available. Inform the user to check with the department."
                
                else:
                    # User might be asking "what classes do I have today?" or similar
                    import datetime
                    
                    # Try to extract day from query, otherwise use today
                    day_mapping = {
                        'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
                        'thursday': 'Thursday', 'friday': 'Friday',
                        'montag': 'Monday', 'dienstag': 'Tuesday', 'mittwoch': 'Wednesday',
                        'donnerstag': 'Thursday', 'freitag': 'Friday'
                    }
                    
                    query_day = None
                    for day_name, day_proper in day_mapping.items():
                        if day_name in query.lower():
                            query_day = day_proper
                            break
                    
                    # If "today" or no day specified, use current day
                    if not query_day or 'today' in query.lower():
                        query_day = datetime.datetime.now().strftime('%A')
                    
                    # Extract semester from query
                    semester_filters = extract_semester_criteria(query)
                    
                    if semester_filters["semester_num"]:
                        # get schedule with semester validation
                        schedule_info, is_active, current_season = get_schedule_for_day(
                            semester_filters["semester_num"], 
                            query_day, 
                            schedule_data
                        )
                        
                        if not is_active:
                            # semester doesn't have classes in current season
                            semester_num = semester_filters["semester_num"]
                            other_season = "Summer" if current_season == "Winter" else "Winter"
                            context = f"""[SEMESTER_MISMATCH]
the student is in semester {semester_num}, which runs in the {other_season} semester.
current season: {current_season}
{current_season} semesters: {', '.join(map(str, [1,3,5,7] if current_season == 'Winter' else [2,4,6]))}

explain this clearly and wish them a good break. be friendly but concise."""
                        elif schedule_info:
                            # has classes today
                            context = f"""[OFFICIAL CLASS SCHEDULE FOR {query_day.upper()}]
semester {semester_filters["semester_num"]} schedule:
{json.dumps(schedule_info, indent=2)}

present this clearly with emojis (📅 day, 🕐 time, 📚 module, 👨‍🏫 professor, 📍 room). include class type."""
                        else:
                            # semester is active but no classes on this day
                            context = f"""[SCHEDULE_INFO]
semester {semester_filters['semester_num']} has no classes on {query_day}. 
tell the student they're free today. be friendly."""
                    else:
                        # No semester specified - ask the user
                        context = "[SYSTEM]: ask which semester they're in to provide the correct schedule."
            
            elif intent == 'modules_list':
                # USER WANTS A LIST OF MODULES FOR THEIR SEMESTER
                filters = extract_semester_criteria(query)
                
                if filters["semester_num"] or filters["season"]:
                    hardcoded_list = get_modules_from_map(module_map, filters)
                    
                    if hardcoded_list:
                        # format for clean presentation
                        context = f"""[MODULE_LIST]
here are all modules for semester {filters.get('semester_num', 'requested')}:

{chr(10).join(hardcoded_list)}

present this as a clean numbered list. no additional commentary unless asked."""
                    else:
                        context = "[SYSTEM]: no modules found for specified criteria."
                else:
                    # User didn't specify semester
                    context = "[SYSTEM]: ask which semester they're in."
            
            elif intent == 'module_info':
                # USER WANTS DETAILED INFO ABOUT A SPECIFIC MODULE
                # Use RAG to retrieve detailed module information
                module_code = find_code_by_name(query, module_map)
                
                if module_code:
                    module_name = module_map.get(module_code, "")
                    
                    # Retrieve detailed module info from RAG
                    retrieved_docs = ensemble_retriever.invoke(query)
                    
                    # Use structured formatter for module details
                    context = format_module_details_from_rag(module_code, module_name, retrieved_docs)
                    
                    # Also check if they want schedule info for this module
                    if any(word in query.lower() for word in ['when', 'schedule', 'time', 'day']):
                        schedule_info = get_schedule_for_module(module_name, schedule_data, module_code)
                        
                        if schedule_info:
                            context += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS SCHEDULE FOR THIS MODULE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{json.dumps(schedule_info, indent=2)}

IMPORTANT: Include this schedule information in your response."""
                else:
                    # Module not found, use general RAG
                    retrieved_docs = ensemble_retriever.invoke(query)
                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            else:
                # GENERAL QUERY - Use RAG retriever
                retrieved_docs = ensemble_retriever.invoke(query)
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            # --- GENERATE RESPONSE ---
            response = generate_chat_response(context, history, query, hardcoded_list)
            print(f"Bot: {response}")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    chat_loop()