import openai
import mysql.connector
from langgraph.graph import StateGraph

openai.api_key = ''# YOUR OPENAI API KEY

# Importazioni necessarie per la creazione degli agenti
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph


import importlib

# Importa i prompt
bot_prompt = importlib.import_module('bot_prompt')

# Ottieni tutte le variabili definite in file1
for attr in dir(bot_prompt):
    # Ignora le variabili speciali e built-in
    if not attr.startswith("__") and isinstance(getattr(bot_prompt, attr), str):
        globals()[attr] = getattr(bot_prompt, attr)


# Funzione per creare un agente con strumenti
def create_agent_tools(llm, tools, system_message: str):
    # Definisce un template di prompt per l'agente
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Imposta i messaggi di sistema e i nomi degli strumenti disponibili
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    # Restituisce l'agente con il prompt e gli strumenti associati
    return prompt | llm.bind_tools(tools)

# Funzione per creare un agente
def create_agent(llm, system_message: str):
    # Definisce un template di prompt per l'agente
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message
                
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Imposta i messaggi di sistema e i nomi degli strumenti disponibili
    prompt = prompt.partial(system_message=system_message)
    # Restituisce l'agente con il prompt e gli strumenti associati
    return prompt | llm

# Definizione degli strumenti (TOOLS)
from langchain_core.tools import tool
from typing import Annotated
import mysql.connector
from langchain_experimental.utilities import PythonREPL

# Inizializzazione di un REPL Python
repl = PythonREPL()

# Strumento per eseguire query SQL
@tool
def query_sql(
    query: Annotated[str, "The SQL query to execute to retrieve data."]
):
    """Executes SQL query and returns JSON."""
    # Connessione al database MySQL
    conn = mysql.connector.connect(
        host="",
        user="",
        passwd="",
        database=""
    )
    cursor = conn.cursor(dictionary=True)
    # Esecuzione della query e recupero dei risultati
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# Strumento per eseguire codice Python
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to modify the data."]
):
    """Executes Python code."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

# Associazione degli strumenti agli agenti
sql_tool = query_sql
python_tool = python_repl

# Definizione della struttura dello stato del grafo
from typing import Annotated, Sequence, TypedDict
import operator

# Definisce la struttura dello stato del grafo
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

from langchain_core.agents import AgentAction, AgentFinish
from typing import  Union
class NewAgentState(TypedDict):

   input: str
   #messaggi degli operatori
   chat_history: list[BaseMessage]
   #represent the most recent agent outcome
   agent_outcome: Union[AgentAction, AgentFinish, None]
   #messaggi completi fino ad ora degli operatori
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
   next: str

# Funzione helper per creare un nodo per un dato agente
import functools

def agent_node(state, agent, name):
    # Log dei messaggi in ingresso
    print(f"Agent '{name}' received messages: {state['messages']}")
    
    # Esegui l'agente con i messaggi forniti
    result = agent.invoke(state)
    
    # Log della risposta dell'agente
    print(f"Agent '{name}' response: {result}")
    
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    return {
        "messages": [result],
        "sender": name,
    }

# Inizializzazione del modello di linguaggio (LLM) con GPT-4
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", openai_api_key="")

# CREAZIONE DEGLI AGENTI

need_data = create_agent(
    llm,
    system_message="",
)
data_node = functools.partial(agent_node, agent=need_data, name="need_data")

sql_agent = create_agent_tools(
    llm,
    [sql_tool],
    system_message=prompt_SQLGenerator, # type: ignore
)
sql_node = functools.partial(agent_node, agent=sql_agent, name="SQLGenerator")

need_code = create_agent(
    llm,
    system_message="",
)
code_node = functools.partial(agent_node, agent=need_code, name="need_code")

python_agent = create_agent_tools(
    llm,
    [python_tool],
    system_message="Generate Python code to process the data.",
)
python_node = functools.partial(agent_node, agent=python_agent, name="CodeGenerator")

FinalGPT = create_agent(
    llm,
    system_message=prompt_FinalGPT, # type: ignore
)
Fanta_node = functools.partial(agent_node, agent=FinalGPT, name="FinalGPT")

# Definizione del nodo per eseguire gli strumenti
from langgraph.prebuilt import ToolNode

tools = [sql_tool, python_tool]
tool_node = ToolNode(tools)

# Definizione della logica di routing

# Either agent can decide to end
from typing import Literal

def sql_router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "NEED_DATA" in last_message.content:
        # Any agent decided the work is done
        return "SQLGenerator"
    return "FinalGPT"

def code_router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "NEED_CODE" in last_message.content:
        # Any agent decided the work is done
        return "CodeGenerator"
    return "FinalGPT"

workflow = StateGraph(AgentState)

workflow.add_node("need_data",data_node )
workflow.add_node("SQLGenerator",sql_node)
workflow.add_node("need_code",code_node )
workflow.add_node("CodeGenerator", python_node)
workflow.add_node("FinalGPT",Fanta_node )
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "need_data",
    sql_router,
    {"SQLGenerator": "SQLGenerator", "FinalGPT": "FinalGPT"},
)

workflow.add_conditional_edges(
    "need_code",
    code_router,
    {"CodeGenerator": "CodeGenerator", "FinalGPT": "FinalGPT"},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "SQLGenerator": "need_code",
        "CodeGenerator": "FinalGPT",
    },
)

# Imposto l'entrata
workflow.set_entry_point("need_data")

workflow.add_edge("SQLGenerator", "call_tool")
workflow.add_edge("CodeGenerator", "call_tool")
workflow.add_edge("FinalGPT", END)

graph = workflow.compile()

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Insert your question here",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")