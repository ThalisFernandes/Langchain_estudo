from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
try:
    if "OPENAI_API_KEY" not in os.environ:
        print("Sem chave de API")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
except Exception as E:
    print(E)


llm = OpenAI(temperature=0,api_key= api_key)


tools = [
    Tool(
        name="CAlculator",
        func=lambda x: eval(x),
        description="Description"
    )
]


agent = initialize_agent(
    tools=tools,
    llm= llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

question = "Qual o valor de 5 * 50"

response = agent.run(question)

print(response)