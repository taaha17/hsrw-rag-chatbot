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
)


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
    global_knowledge = """
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
- Semester 6: Practical internship (can be done abroad)
- Semester 7: Bachelor thesis + colloquium

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
    if "[OFFICIAL CLASS SCHEDULE" in context or "SCHEDULE_INFO" in context:
        # We have schedule data - emphasize using it correctly
        system_prompt = f"""You are the AI Academic Advisor for Infotronic Systems Engineering (ISE) at Rhine-Waal University.

{global_knowledge}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION FOR THIS QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The student is asking about CLASS SCHEDULES. Below you will find OFFICIAL SCHEDULE DATA from our database.

YOUR TASK:
1. READ the schedule data carefully
2. Extract the relevant information (day, time, room, professor)
3. Present it CLEARLY and DIRECTLY to the student
4. DO NOT say "I don't have access" or "check with advisor" - YOU HAVE THE DATA!
5. DO NOT make up or guess any information not in the data

EXAMPLE GOOD RESPONSE:
"Your Signals and Systems class is on Tuesday from 10:00 to 11:30 in Hörsaal with Prof. Dr. Zimmer. It's a lecture (L)."

EXAMPLE BAD RESPONSE:
"I don't have direct access to the schedule. Please check with your advisor." ❌

{list_instruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA PROVIDED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Student Question: {question}
"""
    else:
        # Regular query - use standard prompt
        system_prompt = f"""You are the AI Academic Advisor for Infotronic Systems Engineering (ISE) at Rhine-Waal University.

{global_knowledge}
{list_instruction}

RESPONSE GUIDELINES:
1. Be helpful, friendly, and professional
2. Give direct, accurate answers based on the data provided
3. If you need more information from the student (like their semester), ask politely
4. If the Context has specific data, USE IT - don't tell them to check elsewhere
5. Be concise but complete in your answers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFORMATION AVAILABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Student Question: {question}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    
    data = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, timeout=60)
        return response.json()["message"]["content"]
    except Exception as e:
        return f"❌ Chat Error: {e}"

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
                        schedule_info = get_schedule_for_day(semester_filters["semester_num"], query_day, schedule_data)
                        
                        if schedule_info:
                            context = f"""[OFFICIAL CLASS SCHEDULE FOR {query_day.upper()}]
Semester {semester_filters["semester_num"]} Schedule:
{json.dumps(schedule_info, indent=2)}

IMPORTANT: Present this schedule clearly to the user. Show the time, module name, type (L=Lecture, E=Exercise, P=Practical, etc.), professor, and room."""
                        else:
                            context = f"[SYSTEM]: No classes found for semester {semester_filters['semester_num']} on {query_day}."
                    else:
                        # No semester specified - ask the user
                        context = "[SYSTEM]: User is asking about their schedule but hasn't specified their semester. Ask them: 'Which semester are you in?' so we can provide the correct schedule."
            
            elif intent == 'modules_list':
                # USER WANTS A LIST OF MODULES FOR THEIR SEMESTER
                filters = extract_semester_criteria(query)
                
                if filters["semester_num"] or filters["season"]:
                    hardcoded_list = get_modules_from_map(module_map, filters)
                    
                    if not hardcoded_list:
                        context = "[SYSTEM]: No modules found for the specified criteria. This might be an error - check the module map."
                else:
                    # User didn't specify semester
                    context = "[SYSTEM]: User wants to know their modules but hasn't specified which semester. Ask: 'Which semester are you currently in?' or 'Are you asking about winter or summer semester?'"
            
            elif intent == 'module_info':
                # USER WANTS DETAILED INFO ABOUT A SPECIFIC MODULE
                # Use RAG to retrieve detailed module information
                module_code = find_code_by_name(query, module_map)
                
                if module_code:
                    # Also check if they want schedule info for this module
                    if any(word in query.lower() for word in ['when', 'schedule', 'time', 'day']):
                        module_name = module_map.get(module_code, "")
                        schedule_info = get_schedule_for_module(module_name, schedule_data, module_code)
                        
                        if schedule_info:
                            context = f"""[MODULE INFORMATION + SCHEDULE]
Module: {module_name} ({module_code})

Schedule:
{json.dumps(schedule_info, indent=2)}

IMPORTANT: Answer using this schedule data. Also retrieve any additional module details from the RAG database."""
                
                # Use RAG retriever to get detailed module info
                retrieved_docs = ensemble_retriever.invoke(query)
                rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                
                if context:
                    context += f"\n\nAdditional Module Details:\n{rag_context}"
                else:
                    context = rag_context
            
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