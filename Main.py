from Frontend.GUI import(
    GraphicalUserInterface,
    SetAssistantStatus,
    ShowTextToScreen,
    TempDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    QueryModifier,
    GetAssistantStatus,
    GetMicrophoneStatus)
from Backend.RealtimeSearchEngine import RealtimeSearchEngine
from Backend.SpeechToText import SpeechRecognition
from Backend.TextToSpeech import TextToSpeech
from Backend.EmotionHandler import EmotionalDecisionMaker  # Updated import
from dotenv import dotenv_values
from time import sleep
import threading
import json
import os

env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
    
def ShowDefaultChatIfNoChats():
    Assistantname = env_vars.get("Assistantname")
    DefaultMessage = f'''{Username} : Hello {Assistantname}, How are you?
    {Assistantname}: Welcome {Username}. I am doing well. How may i help you?'''
    File = open(r'Data\Chatlog.json',"r",encoding='utf-8')
    if len(File.read())<5:
        with open(TempDirectoryPath('Database.data'),'w',encoding='utf-8') as file:
            file.write("")
                
        with open(TempDirectoryPath('Responses.data'), 'w', encoding='utf-8') as file:
            file.write(DefaultMessage)

def ReadChatLogJson():
    with open(r'Data\Chatlog.json', 'r' , encoding='utf-8') as file:
        chatlog_data = json.load(file)
    return chatlog_data

def ChatLogIntegration():
    Assistantname = env_vars.get("Assistantname")
    json_data = ReadChatLogJson()
    formatted_chatlog = ""
    for entry in json_data:
        if entry["role"] == "user":
            formatted_chatlog += f"User: {entry['content']}\n"
        elif entry["role"] =="assistant":
            formatted_chatlog += f"User: {entry['content']}\n"
    formatted_chatlog = formatted_chatlog.replace("User",Username + " ")
    formatted_chatlog = formatted_chatlog.replace("Assistant",Assistantname + " ")
    
    with open(TempDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted_chatlog))
        
def ShowChatsOnGUI():
    File = open(TempDirectoryPath('Database.data'), "r", encoding='utf-8')
    Data = File.read()
    if len(str(Data))>0:
        lines = Data.split('\n')
        result = '\n'.join(lines)
        File.close()
        File = open(TempDirectoryPath('Responses.data'), "w", encoding='utf-8')
        File.write(result)
        File.close()
        
def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatsOnGUI()

InitialExecution()

def MainExecution():
    Assistantname = env_vars.get("Assistantname")
    emotion_handler = EmotionalDecisionMaker()  # Create instance

    SetAssistantStatus("Listening...")
    Query = SpeechRecognition()
    ShowTextToScreen(f"{Username}: {Query}")
    
    SetAssistantStatus("Thinking...")
    analysis = emotion_handler.analyze_query(Query)  # Get analysis directly
    
    if analysis['query_type'] == 'realtime':
        SetAssistantStatus("Searching...")
        Answer = RealtimeSearchEngine(QueryModifier(Query))
        response = Answer
    else:
        response, _ = emotion_handler.process_query(Query)
        
    ShowTextToScreen(f"{Assistantname} ({analysis['emotion']}): {response}")
    SetAssistantStatus("Answering...")
    # Pass emotion parameters to TextToSpeech
    TextToSpeech(response, emotion_params={
        'emotion': analysis['emotion'],
        'confidence': analysis['confidence'],
        'sentiment': analysis['sentiment']
    })
    return True

def FirstThread():
    while True:
        CurrentStatus = GetMicrophoneStatus()
        if CurrentStatus == "True":
            MainExecution()
        else:
            AIStatus = GetAssistantStatus()
            if "Available..." in AIStatus:
                sleep(0.1)
            else:
                SetAssistantStatus("Available...")

def SecondThread():
    GraphicalUserInterface()

if __name__ == "__main__":
    thread2 = threading.Thread(target=FirstThread, daemon=True)
    thread2.start()
    SecondThread()




