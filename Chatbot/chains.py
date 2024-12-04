from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.base import Runnable
from pydantic import BaseModel, Field

contextual_q_system_prompt = (
  "Given a chat history and the latest user question "
  "which might reference context in the chat history, "
  "formulate a standalone question which can be understood "
  "without the chat history. Do NOT answer the question, "
  "just reformulate it if needed and otherwise return it as is."
)

class Recommendations(BaseModel):
  recommendations: list[dict] = Field(
    default=[],
    description=(
      "List of course recommendations. Each item should include:"
      " 'course_id': A string representing the course identifier,"
      " 'description': A concise description of the course,"
      " 'why': A brief explanation of why this course is recommended."
    )
  )
  
def createRecommendationChain(model, model_mini, retriever) -> Runnable:
  contextual_q_prompt = ChatPromptTemplate.from_messages([
      ("system", contextual_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      MessagesPlaceholder("user_info"),
      ("human", "{input}")
  ])
  history_aware_retriever = create_history_aware_retriever(
      model_mini, retriever, contextual_q_prompt
  )
  
  system_prompt = """
    You are a supportive and friendly assistant dedicated to helping foreign students adapt to university life. 
    Your goal is to provide clear, accurate, and actionable course recommendations. 
    Based on the provided context and user query, generate a structured JSON response with the following format:

    {{
      "recommendations": [                 # List of course recommendations.
        {{
          "course_id": "string",   # Course identifier.
          "description": "string", # Concise course description.
          "why": "string"          # Reason for recommending this course.
        }},
        ...
      ]
    }}

    If the context or query does not allow meaningful recommendations, set "return_value" to false and leave "content" as an empty list.

    <context>
    {context}
    </context>
  """
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("user_info"),
    ("human", "{input}")
  ])
  parser = JsonOutputParser(pydantic_object=Recommendations)
  qa_chain = create_stuff_documents_chain(model, prompt, output_parser=parser)
  
  return create_retrieval_chain(history_aware_retriever, qa_chain)

def createAdviceChain(model, model_mini, retriever) -> Runnable:
  contextual_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextual_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("user_info"),
    ("human", "{input}")
  ])
  history_aware_retriever = create_history_aware_retriever(
    model_mini, retriever, contextual_q_prompt
  )

  system_prompt = """
    You are a supportive and friendly assistant dedicated to helping foreign students adapt to university life. 
    Your goal is to provide clear, accurate, and actionable information about academic guidance, campus resources, 
    and general advice. Always respond in a concise and positive tone, tailoring your answers to the user’s needs. 

    Answer the user's questions using the context provided below. If the context lacks sufficient information, 
    do not speculate or provide inaccurate answers. Instead, politely say, "I don't know," and, if possible, 
    suggest how the user might find the required information.

    <context>
    {context}
    </context>
  """
  qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("user_info"),
    ("human", "{input}")
  ])
  qa_chain = create_stuff_documents_chain(model, qa_prompt)
  
  return create_retrieval_chain(history_aware_retriever, qa_chain)

def createAnalyzeChain(model) -> Runnable:
  system_prompt = """
    You are an intelligent assistant responsible for determining the user's intent.
    Identify the intent in one of these categories:

    - "recommendation"
    - "advice"
    - "information"
    - "general"

    1. **Triggering "information"**:
      - `information` should only trigger when the input contains explicit user-specific data and is **not a question**.
      - Examples: `"My GPA is 3.5."`, `"<제1전공 공통교과>
            이수학기 과목번호 교과목명 학점 성적 비고
            2019-1 STS2005 미적분학I 3.0 B- AB
            2019-1 COR1015 알바트로스세미나 1.0 S S/U
            2019-1 COR1011 컴퓨팅사고력(고급) 3.0 C0 AB
            2019-1 COR1013 자연계글쓰기 3.0 C+ AB
            2019-2 COR1003 영어글로벌의사소통I 3.0 A0 E , AB
            2019-2 CHS2004 현대서양의형성 3.0 B0
            2022-1 SHS2002 한국과세계 3.0 A0
            2022-1 COR1007 성찰과성장 1.0 S T , S/U
            2022-2 HFS2001 철학적인간학 3.0 C+
            소요학점 23 학점 이수학점 23.0 학점"`
      - Questions should not be classified as `information`.

    2. **Triggering "advice"**:
      - Queries about academic or administrative support (e.g., course registration, graduation requirements, visa applications) should always be classified as `advice`.
      - Examples: `"What is the deadline for course registration?"`, `"How do I apply for a visa?"`

    3. **Triggering "general"**:
      - Queries referencing prior chat history or generic questions unrelated to user data or academic topics should be classified as `general`.
      - Examples: `"What did I ask about earlier?"`, `"Translate 'hello' into Korean."`

    4. **Triggering "recommendation"**:
      - Queries explicitly seeking course recommendations should always be classified as `recommendation`.
      - Example: `"What courses do you recommend for AI?"`

    5. **Fallback Rule**:
      - If the intent is unclear or ambiguous, default to `general`.
    
    Return only the intent as a single lowercase word.
  """
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
  ])
  
  return prompt | model

class UserInfo(BaseModel):
  preferences: list[str] = Field(
    default=[], description="List of user preferences or an empty list if not requested."
  )
  career_goals: list[str] = Field(
    default=[], description="List of user career goals or an empty list if not requested."
  )
  academic_status: str = Field(
    default="", description="Current academic status or an empty string if not requested."
  )
  graduation_requirements: dict = Field(
    default={
        "major_requirements": False,
        "core_courses": False,
        "total_credits": False,
    },
    description="Graduation requirement status for major, core courses, and total credits.",
  )
  total_credits: int = Field(
    default=0, description="Total earned credits or 0 if not requested."
  )
  gpa: float = Field(
    default=0.0, description="GPA as a float or 0.0 if not requested."
  )
  core_courses_completed: dict = Field(
    default={
        "intro_courses": False,
        "mandatory_courses": False,
    },
    description="Completion status for intro and mandatory core courses.",
  )
  completed_courses: list[str] = Field(
    default=[], description="List of completed courses or an empty list if not requested."
  )
    
def createUserInfoChain(model) -> Runnable:
  system_prompt = """
    You are an intelligent assistant responsible for providing user information. 
    Based on the user's input and existing user information, return the requested details in JSON format.

    If the user's query explicitly asks for specific information (e.g., preferences, career goals, completed courses, etc.), 
    return only the relevant fields. Use the following structure:

    {{
      "preferences": [list of user preferences, or empty list if not requested],
      "career_goals": [list of user career goals, or empty list if not requested],
      "academic_status": "current academic status, or an empty string if not requested",
      "graduation_requirements": {{
        "major_requirements": true/false,
        "core_courses": true/false,
        "total_credits": true/false
      }},
      "total_credits": total earned credits, or 0 if not requested,
      "gpa": GPA as a float, or 0.0 if not requested,
      "core_courses_completed": {{
        "intro_courses": true/false,
        "mandatory_courses": true/false
      }},
      "completed_courses": [list of completed courses, or empty list if not requested]
    }}

    If the user's query does not specify any particular information, return the entire userInfo object.

    <user_info>
    {user_info}
    </user_info>
  
  """
  
  parser = JsonOutputParser(pydantic_object=UserInfo)
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("user_info"),
    ("human", "{input}")
  ])
  
  return prompt | model | parser

def createGeneralChain(model) -> Runnable:
  system_prompt = """
    You are an intelligent assistant designed to provide general responses to user queries. 
    Your goal is to respond accurately and concisely to any input without relying on prior chat history or user information. 
    Always provide relevant and straightforward answers.

    If the input is ambiguous or cannot be answered directly, politely ask the user for clarification.
  """
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("user_info"),
    ("human", "{input}")
  ])
  
  return prompt | model
  