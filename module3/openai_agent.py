from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from pprint import pprint
# Building graph with MemorySaver checkpointer
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


model = ChatOpenAI(model = "gpt-4o", temperature=0)
from langchain_core.messages import AnyMessage,BaseMessage
from pydantic import BaseModel, Field
from typing import List,Sequence, Annotated,Optional
from langgraph.graph.message import add_messages
class State(BaseModel):
    messages:  Annotated[List[AnyMessage], add_messages]= Field(description="List of messages between agent and user")
    summary: Optional[str] = Field(None,description="Summary of the message history so far")
    
    
# define logic o call model
def call_model(state:State):
    state_dict= state.model_dump()
    summary = state_dict.get("summary", "")
    if summary:
        
        # add summary to the system message
        system_message = f"Summary of the conversation earlier: {summary}"
        
        # append summary to newer messages
        messages = [SystemMessage(content=system_message)] + state_dict["messages"]
    else:
        messages = state_dict["messages"]
    
    response = model.invoke(messages)
    return {"messages":response}        
    

# message summary node
def summarize_conversation(state: State):
    state_dict = state.model_dump()
    summary = state_dict.get("summary", "")
    
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    # add prompt to our history
    messages = state_dict["messages"] + [HumanMessage(content = summary_message)]
    response = model.invoke(messages)
    print(f"in summary node: {response}")
    
    # delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m["id"]) for m in state_dict["messages"][:-2]]
    return {"summary": response.content, "messages":delete_messages}
    

def should_continue(state:State):
    """ 
    Return to the next node to execute

    Args:
        state (State): Current state of the graph
    """
    messages = state.model_dump()["messages"]
    if len(messages)>6: # summarize conversaiton if we have >6 messages
        return "summarize_conversation"
    
    # otherwise we can just end it
    return END

def build_graph():
    # build graph
    workflow = StateGraph(State)
    # add nodes
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)
    # add edges
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)
    # add memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
#display
graph = build_graph()
display(Image(graph.get_graph().draw_mermaid_png()))