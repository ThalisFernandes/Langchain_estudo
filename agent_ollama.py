from langchain_community.llms import Ollama 
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.cache import InMemoryCache
from langchain.globals import  set_llm_cache

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


from cache_wrapper import SemanticCacheWrapper

""" embedding_function = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="cache",
    embedding_function=embedding_function,
    persist_directory="chromadb_cache"
)
 """

# Define the llm model using Ollama mistral
llm = Ollama(model="mistral")

cached_llm = SemanticCacheWrapper(llm=llm)



response = cached_llm.invoke("Qual a capital do Brasil?")

#cache = SemanticCache.from_llm(
#    llm=llm,
#    vectorstore= vector_store,
#    score_threshold=0.8
#)


#Define the llm memory
# set_llm_cache(cache)

#Creating a simple tool, to solve problems with the AI agent
""" 
tools = [
    Tool(
        name="CAlculator",
        func=lambda x: eval(x),
        description="Description"
    )
]
 """

# Agent initialization, 
""" agent = initialize_agent(
    tools=tools,
    llm= llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
) """

""" question = "Qual o valor de 5 * 50"

response = agent.run(question)

print(response)

 """