from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import os 

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class Exam(BaseModel):
    defination:str =Field(description="give deatailed defination of the topic")
    keywords:str = Field(description="give keywords of the topic ")
    exam_format:str = Field(description="give a exam written format ")
    marks:str = Field(description="give marks wise answers for topic like 2m , 4m , 8m")
    gunpoints:str = Field(description="give gunpoints for the topic ")
    quick_revision:str = Field(description="give a quick revision format for the topic")


parcer = PydanticOutputParser(
    pydantic_object=Exam
)

prompt = ChatPromptTemplate.from_messages([
    "system",
    """act as a senior exam paper corector ,
    give output in a required format.
    {format_instructions}""",
    MessagesPlaceholder("history"),
    ("human"),("{input}")]
)

chain = prompt.partial(
    format_instructions=parcer.get_format_instructions
    )|llm|parcer


store = {}
def get_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

mentor = RunnableWithMessageHistory(
    
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("===============================AI=======================================")

while True:
    user_input = input("\nyou")

    if user_input.lower == "exit":
        break

    response = mentor.invoke({
        "input":user_input
    },
    config={
        "configurable":{"session_id":"sandeep"}
    })


    print("\ndefination: ",response.defination)
    print("\nkeywords: ",response.keywords)
    print("\nexam_format ",response.exam_format)
    print("\nmarks_wise: ",response.marks)
    print("\ngunpoints: ",response.gunpoints)
    print("\nquick_revision: ",response.quick_revision)