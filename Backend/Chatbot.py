from groq import Groq #importing groq library to use this api
from json import load, dump #importing functions to read and write json files
import datetime #importing the datetime module for real - time date and time information
from dotenv import dotenv_values #importing dotenv_values to read environment varibales from a .env file
from Backend.EmotionHandler import process_emotional_response

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
#initialize an empty list to store chat messages
messages = []

#define a system message that provides context to the ai chatbot about its role and behaviour
system = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

#a list of system instructions for the chatbot
SystemChatBot = [
    {"role": "system", "content": system}
]

#attempt to load the chat log from a json file
try:
    with open(r"Data\ChatLog.json","r") as f:
        messages = load(f) #load existing messages from the chat log
except FileNotFoundError:
    #if the file doesnot exist , create an empty json file to store chat logs
    with open(r"Data\ChatLog.json","w") as f:
        dump([], f)
        
#function to get real-time date and time information
def RealtimeInformation():
    current_date_time = datetime.datetime.now() #get the current date amnd time
    day = current_date_time.strftime("%A") #day of the week
    date = current_date_time.strftime("%d") #day of the month
    month = current_date_time.strftime("%B") #full month
    year = current_date_time.strftime("%Y") #year
    hour = current_date_time.strftime("%H") #hour in 24-hour format
    minute = current_date_time.strftime("%M") #minute
    second = current_date_time.strftime("%S") #second
    
    #format the information into a string
    data = f"please use this real-time infromation if needed,\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time:{hour} hours :{minute} minutes:{second} seconds.\n"
    return data

#function to modify the chatbot's response for better formatting
def AnswerModifier(Answer):
    lines = Answer.split('\n') #split the response into lines
    non_empty_lines = [line for line in lines if line.strip()] #remove empty lines
    modified_answer = '\n'.join(non_empty_lines) #join the cleaned lines back together
    return modified_answer

#main chatbot function to handle queries
def ChatBot(query, emotion_context=None):
    """This function sends the user's query  to the chatbot and returns the AI's response. """
    
    try :
        #load the existing chat log from the json file.
        with open(r"Data\ChatLog.json","r") as f:
            messages = load(f)
            
        #append the user's query to the message list
        messages.append({"role": "user", "content": f"{query}"})
        
        # Add emotional context to system message
        system_msg = system
        if emotion_context:
            system_msg += f"\nUser's current emotional state: {emotion_context['emotion']} (confidence: {emotion_context['confidence']:.2f})"
            
        #make a request to the Groq API for a response
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        
        base_response = "" #initialise an empty string to store AI's response
        
        # process the streamed response chunks
        for chunk in completion:
            if chunk.choices[0].delta.content:
                base_response += chunk.choices[0].delta.content
        
        base_response = base_response.replace("</s>", "") #clean up any unwanted token from the response
        
        # Process response with emotion handling
        emotional_response, emotion_data = process_emotional_response(query, base_response)
        
        #append the chatbot's response to messages list
        messages.append({"role":"assistant","content":emotional_response})
        
        #save the chatbot's response to the messages list
        with open(r"Data\ChatLog.json","w") as f:
            dump(messages, f , indent=4)
            
        #return the formatted response
        return emotional_response, emotion_data
    
    except Exception as e:
        #handle errors by printing the exception and resulting the chatlog
        print(f"Error:{e}")
        with open(r"Data\Chatlog.json", "w") as f:
            dump([], f, indent=4)
        return "I apologize, but I'm having trouble processing that right now.", None

#main program entry point
if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Question: ") #prompt the user for a question
        print(ChatBot(user_input)) #call the chatbot function and print its response
