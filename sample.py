import asyncio
from pdfminer.high_level import extract_text
from Chatbot.chatbot import Chatbot 
import uuid 

# Define test cases
test_cases = [
  # 강의 추천
  "I'm interested in artificial intelligence and machine learning. What courses should I take?",
  "Are there any advanced courses in computer systems for next semester?",
  
  # 학사 정보 제공
  "How many credits do I need to graduate?",
  "What is the GPA requirement for a scholarship at Sogang University?",
  "Can you provide information about the visa extension process?",
  "What are the required core courses for computer science majors?",
  
  # 사용자 데이터 관련
  "What is my current GPA?",
  "Can you list the courses I have already completed?",
  "Have I completed the required core courses for graduation?",
  "What are my preferences for future courses?",
  
  # 일반적인 질문
  "How do I join a student club at Sogang University?",
  "What is the difference between CSE4130 and CSE4185?",
  "Can you translate '학생증 발급' into English?",
  "What is the deadline for course registration this semester?",
  
  # 기억 기능 테스트
  "What did I ask about earlier regarding core courses?",
  "Can you summarize the topics we discussed today?",
]

async def run_tests():
  chatbot = Chatbot()
  
  pdf_path = "user_data.pdf"  # data path of my 과목이수표 
  user_data = extract_text(pdf_path)
  session_id = str(uuid.uuid4())

  # Open a text file to save the results
  output_file = "test_results.txt"
  with open(output_file, "w") as f:
    f.write("Chatbot Test Results\n")
    f.write("=" * 50 + "\n")
    
    query = "i am intersted in AI and computer system and i want to be expert on neural net" + user_data
    result = await chatbot.ainvoke(query, session_id)
    
    f.write(f"Test Case 0:\n")
    f.write(f"Query: provide user data\n")
    f.write(f"Result: {result}\n")
    f.write("=" * 50 + "\n")
    
    # Process each test case
    for idx, query in enumerate(test_cases, start=1):
      try:
        result = await chatbot.ainvoke(query, session_id)

        f.write(f"Test Case {idx}:\n")
        f.write(f"Query: {query}\n")
        f.write(f"Result: {result}\n")
        f.write("=" * 50 + "\n")

      except Exception as e:
        f.write(f"Test Case {idx}:\n")
        f.write(f"Query: {query}\n")
        f.write(f"Error: {str(e)}\n")
        f.write("=" * 50 + "\n")

# Run the tests
if __name__ == "__main__":
  asyncio.run(run_tests())
