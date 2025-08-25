from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import configuration

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

# chatbot instruction
MODEL_SYSTEM_PROMPT = """ 
You are a helpful assistant with memory that provides information about user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}
"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_PROMPT = """ 
You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully.
2. Identify new information about the user such as:
    - Personal information (name, location, phone number, email, linkedin profile, github profile, portfolio, etc.)
    - Professional Summary(1-2 sentences)
    - Education (Degree, School, Graduation Year)
    - Work Experience (jobtitle, company, dates, responsibilities, links)
    - Publications (title, date, journal, link)
    - Technical Skills (programming languages, frameworks, tools, any specific skills focused on things like (AI, Machine Learning, Data Science, etc.))
    - Projects (title, description, skills, links)
    - Preferences (likes, dislikes)
    - Interests and hobbies
    - Goals or future plans
3. Merge any new information with the existing memory.
4. Format the memory as clear, bulleted list.
5. If the information conflicts with existing memory, keep the most recent version.

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences

Based on the chat history below, please update the user history.
"""

def call_model(state:MessagesState, config:RunnableConfig, store:BaseStore):
    """ Load memory from store and personalize the chatbot's response """
    
    # get user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # Retrieve memory from store
    namespace = ('memory', user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    
    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found"
    
    # format the memory
    system_message = MODEL_SYSTEM_PROMPT.format(memory=existing_memory_content)
    
    # respond using memory as well as the chat history
    response = llm.invoke([SystemMessage(content=system_message)] + state['messages'])
    
    return {'messages': [response]}

def write_memory(state:MessagesState, config:RunnableConfig, store:BaseStore):
    """ Create new memory from the chat history and any existing memory """
    
    # get user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # retrieve existing memory
    namespace = ('memory', user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    
    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found"

    system_message = CREATE_MEMORY_PROMPT.format(memory=existing_memory_content)
    new_memory = llm.invoke([SystemMessage(content=system_message)] + state['messages'])
    
    store.put(namespace, key, {"memory": new_memory.content})   

# build graph
# Define the graph
builder = StateGraph(MessagesState,config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()
