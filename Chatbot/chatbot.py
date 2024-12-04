import os
import logging
from typing import Sequence, TypedDict, Literal
from typing_extensions import Annotated
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from dotenv import load_dotenv
import json

from .loader import Loader
from .retriever import createRetriever
from .chains import (
  createAdviceChain,
  createGeneralChain,
  createRecommendationChain,
  createUserInfoChain,
  createAnalyzeChain,
)

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(message)s",
  level=logging.DEBUG,  
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

class State(TypedDict):
  input: str
  chat_history: Annotated[Sequence[BaseMessage], add_messages]
  context: str
  user_info: Annotated[Sequence[BaseMessage], add_messages]
  answer: str
  intent: str

class Chatbot:
  def __init__(self):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    logging.info("Initializing Chatbot...")
    self._initialize_models()
    self._initialize_chains()
    self._setup_workflow()

  def _initialize_models(self):
    logging.info("Initializing models...")
    portkey_headers = createHeaders(
      api_key=PORTKEY_API_KEY,
      provider="openai",
    )
    self.model = ChatOpenAI(
      base_url=PORTKEY_GATEWAY_URL,
      default_headers=portkey_headers,
      model="gpt-4o",
      temperature=0.3
    )
    self.model_mini = ChatOpenAI(
      base_url=PORTKEY_GATEWAY_URL,
      default_headers=portkey_headers,
      model="gpt-4o-mini",
      temperature=0.3
    )
    logging.info("Models initialized successfully.")

  def _initialize_chains(self):
    logging.info("Initializing chains...")
    loader = Loader()
    docs = loader.invoke("Data")
    logging.debug(f"Loaded {len(docs)} documents for retriever.")
    retriever = createRetriever(self.model, docs)

    self.analyze: Runnable = createAnalyzeChain(self.model)
    self.general: Runnable = createGeneralChain(self.model_mini)
    self.advice: Runnable = createAdviceChain(self.model, self.model_mini, retriever)
    self.recommendation: Runnable = createRecommendationChain(self.model, self.model_mini, retriever)
    self.userInfo: Runnable = createUserInfoChain(self.model)

    logging.info("Chains initialized successfully.")

  def _setup_workflow(self):
    logging.info("Setting up workflow...")

    workflow = StateGraph(state_schema=State)
    memory = MemorySaver()

    workflow.add_node("analyze", self._analyze)
    workflow.add_node("advice", self._advice)
    workflow.add_node("recommendation", self._recommendation)
    workflow.add_node("general", self._general)
    workflow.add_node("information", self._userInfo)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges(
      "analyze",
      self.route,
    )
    self.app = workflow.compile(checkpointer=memory)

    logging.info("Workflow setup completed.")

  async def route(self, state: State) -> Literal["recommendation", "advice", "general", "information"]:
    intent = state['intent']
    logging.info(f"Routing based on intent: {intent}")
    return intent

  async def _analyze(self, state: State):
    logging.info("Starting analyze...")
    response: AIMessage = await self.analyze.ainvoke(state)
    logging.debug(f"Analyzer response: {response.content}")
    return {
      "intent": response.content,
    }

  @staticmethod
  def to_AIMsg_list(user_info: dict) -> list[BaseMessage]:
    logging.debug(f"Converting user_info to AIMessage list: {user_info}")
    return [AIMessage(content=f"{key}: {value}") for key, value in user_info.items()]

  async def _userInfo(self, state: State):
    logging.info("Starting userInfo processing...")
    response = await self.userInfo.ainvoke(state)
    logging.debug(f"userInfo response: {response}")
    return {
      "user_info": Chatbot.to_AIMsg_list(response),
      "answer": response,
      "intent": state['intent']
    }

  async def _recommendation(self, state: State):
    logging.info("Starting recommendation...")
    response: AIMessage = await self.recommendation.ainvoke(state)
    logging.debug(f"Recommendation response: {response}")
    return {
      "chat_history": [
        HumanMessage(content=state['input']),
        AIMessage(content=json.dumps(response["answer"], indent=2))
      ],
      "context": response['context'],
      "answer": response["answer"],
      "user_info": state['user_info'],
      "intent": state['intent']
    }

  async def _advice(self, state: State):
    logging.info("Starting advice...")
    response = await self.advice.ainvoke(state)
    logging.debug(f"Advice response: {response}")
    return {
      "chat_history": [
        HumanMessage(content=state["input"]),
        AIMessage(content=response["answer"]),
      ],
      "context": response["context"],
      "answer": response["answer"],
      "user_info": state["user_info"],
      "intent": state['intent']
    }

  async def _general(self, state: State):
    logging.info("Starting general query processing...")
    response = await self.general.ainvoke(state)
    logging.debug(f"General response: {response.content}")
    return {
      "chat_history": [
        HumanMessage(content=state["input"]),
        AIMessage(content=response.content),
      ],
      "answer": response.content,
      "user_info": state["user_info"],
      "intent": state['intent']
    }

  async def ainvoke(self, query, session_id):
    try:
      logging.info(f"Processing query for session: {session_id}")
      result = await self.app.ainvoke(
        {"input": query, "chat_history": [], "user_info": []},
        config={"configurable": {"thread_id": session_id}},
      )
      logging.info(f"Query processed successfully for session {session_id}")
      return {
        "intent": result["intent"],
        "answer": result["answer"],
      }
    except Exception as e:
      logging.error(f"Error during query processing: {e}")
      raise
