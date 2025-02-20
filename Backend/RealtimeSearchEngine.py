from googlesearch import search
from groq import Groq
from json import load , dump
import datetime
from dotenv import dotenv_values

#load environmental variables from the .env file
env_vars = dotenv_values(".env")

#retrieve specific environment variables for username , assistant name , and API key
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

#initialize the Groq client using the provided API key
try:
    client = Groq(api_key=GroqAPIKey)
except TypeError as e:
    print("Error initializing Groq client:", e)
    exit()

#define the system instruction for the chatbot
system = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""

try:
    with open(r"Data\Chatlog.json", "r") as f:
        messages = load(f)
except:
    with open(r"Data\Chatlog.json", "w") as f:
        dump([], f)
        
        
def GoogleSearch(query):
    results = list(search(query, num_results=5))
    Answer = f"The search results for '{query}' are:\n[start]\n"
    for result in results:
        Answer += f"Title: {result}\n"
    Answer += "[end]"
    return Answer

def AnswerModifier(Answer):
    lines = Answer.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

SystemChatBot = [
    {"role": "system", "content": system},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello , how can I help you?"}
]

def Information():
    data=""
    current_date_time = datetime.datetime.now() #get the current date amnd time
    day = current_date_time.strftime("%A") #day of the week
    date = current_date_time.strftime("%d") #day of the month
    month = current_date_time.strftime("%B") #full month
    year = current_date_time.strftime("%Y") #year
    hour = current_date_time.strftime("%H") #hour in 24-hour format
    minute = current_date_time.strftime("%M") #minute
    second = current_date_time.strftime("%S") #second
    data += f"Use This Real-Time Information if need:\n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second} seconds.\n"
    return data


def RealtimeSearchEngine(prompt):
    global SystemChatBot, messages
    
    #load the existing chat log from the json file.
    with open(r"Data\ChatLog.json","r") as f:
        messages = load(f)
        
    #append the user's query to the message list
    messages.append({"role": "user", "content": f"{prompt}"})
    
    SystemChatBot.append({"role":"user" , "content":GoogleSearch(prompt)})
    
    # generate respond using groq client
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=SystemChatBot + [{"role": "system", "content": Information()}] + messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=1,
        stream=True,
        stop=None
    )
    
    Answer = ""
    
    for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
                
    Answer = Answer.strip().replace("<s/>", "")
    messages.append({"role":"assistant","content":Answer})
    
    with open(r"Data\ChatLog.json","w") as f:
            dump(messages, f , indent=4)
            
            
    #remove the most recent system message from the chatbot conversation
    SystemChatBot.pop()
    return AnswerModifier(Answer=Answer)

#main entry point of the program for interaction
if __name__ == "__main__":
    while True:
        prompt = input("Enter Your query: ")
        print(RealtimeSearchEngine(prompt))



