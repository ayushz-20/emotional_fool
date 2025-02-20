from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import numpy as np
from typing import Tuple, Dict
from groq import Groq
from datetime import datetime
from json import load, dump
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")

class EmotionalDecisionMaker:
    def __init__(self):
        # Initialize emotion classifier with the specified model
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 
                             'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
                             'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
                             'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
                             'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=GroqAPIKey)
        
        # Load response templates
        self.load_response_templates()
        
        # Load chat history
        try:
            with open(r"Data\ChatLog.json", "r") as f:
                self.messages = load(f)
        except FileNotFoundError:
            self.messages = []
            with open(r"Data\ChatLog.json", "w") as f:
                dump(self.messages, f)

        # Keywords for realtime detection
        self.realtime_keywords = [
            'current', 'latest', 'now', 'today', 'news',
            'weather', 'price', 'stock', 'live', 'update'
        ]

    def load_response_templates(self):
        """Load emotional response templates"""
        self.templates = {
            'joy': ["With happiness, {response}", "I'm glad to tell you that {response}"],
            'sadness': ["I understand this is difficult. {response}", "Let me help you through this. {response}"],
            'anger': ["I hear your frustration. {response}", "Let's address this calmly. {response}"],
            'fear': ["Stay calm, {response}", "Don't worry, {response}"],
            'surprise': ["Remarkably, {response}", "Interestingly, {response}"],
            'love': ["With warmth, {response}", "I'm touched to share that {response}"],
            'neutral': ["{response}", "Here's what I found: {response}"]
        }

    def detect_emotion(self, text: str) -> Dict:
        """Detect emotion using the emotion-english-distilroberta-base model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top 3 emotions
        top_k = 3
        top_k_values, top_k_indices = torch.topk(predictions[0], top_k)
        
        emotions = []
        for i in range(top_k):
            emotions.append({
                'emotion': self.emotion_labels[top_k_indices[i]],
                'score': float(top_k_values[i])
            })
        
        # Get sentiment using TextBlob as additional context
        sentiment = TextBlob(text).sentiment.polarity
        
        return {
            'primary_emotion': emotions[0]['emotion'],
            'emotion_score': emotions[0]['score'],
            'secondary_emotions': emotions[1:],
            'sentiment': sentiment
        }

    def analyze_query(self, query: str) -> Dict:
        """Analyze both emotion and query type"""
        # Get emotional analysis
        emotion_analysis = self.detect_emotion(query)
        
        # Determine if query needs realtime information
        is_realtime = any(keyword in query.lower() for keyword in self.realtime_keywords)
        
        return {
            'query_type': 'realtime' if is_realtime else 'general',
            'emotion': emotion_analysis['primary_emotion'],
            'confidence': emotion_analysis['emotion_score'],
            'secondary_emotions': emotion_analysis['secondary_emotions'],
            'sentiment': emotion_analysis['sentiment']
        }

    def format_response(self, response: str, emotion: str, confidence: float) -> str:
        """Format response based on detected emotion"""
        if confidence < 0.3:
            emotion = 'neutral'
            
        # Map similar emotions to template categories
        emotion_mapping = {
            'joy': ['joy', 'amusement', 'excitement', 'gratitude', 'optimism', 'pride'],
            'sadness': ['sadness', 'disappointment', 'grief', 'remorse'],
            'anger': ['anger', 'annoyance', 'disgust', 'disapproval'],
            'fear': ['fear', 'nervousness'],
            'surprise': ['surprise', 'realization'],
            'love': ['love', 'caring', 'admiration'],
            'neutral': ['neutral', 'confusion', 'curiosity']
        }
        
        # Find the appropriate template category
        template_category = 'neutral'
        for category, emotions in emotion_mapping.items():
            if emotion in emotions:
                template_category = category
                break
                
        template_list = self.templates.get(template_category, self.templates['neutral'])
        selected_template = np.random.choice(template_list)
        return selected_template.format(response=response)

    def get_realtime_context(self) -> str:
        """Get current date/time context"""
        now = datetime.now()
        return (f"Current context - Date: {now.strftime('%Y-%m-%d')}, "
                f"Time: {now.strftime('%H:%M:%S')}, "
                f"Day: {now.strftime('%A')}")

    def process_query(self, query: str) -> Tuple[str, Dict]:
        """Main function to process query and generate response"""
        # Make this method synchronous since we're not using async features
        analysis = self.analyze_query(query)
        
        # Prepare system message based on query type
        system_msg = (
            "You are an AI assistant responding to realtime queries. "
            "Keep responses current and factual." if analysis['query_type'] == 'realtime'
            else "You are an empathetic AI assistant. Consider the user's emotional state "
                 f"({analysis['emotion']}) in your response."
        )

        # Add query to messages
        self.messages.append({"role": "user", "content": query})

        # Get response from Groq
        completion = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "system", "content": self.get_realtime_context()},
                *self.messages
            ],
            max_tokens=1024,
            temperature=0.7
        )

        base_response = completion.choices[0].message.content

        # Format response with emotional context
        emotional_response = self.format_response(
            base_response, 
            analysis['emotion'], 
            analysis['confidence']
        )

        # Save to chat history
        self.messages.append({"role": "assistant", "content": emotional_response})
        with open(r"Data\ChatLog.json", "w") as f:
            dump(self.messages, f, indent=4)

        return emotional_response, analysis

def get_response(query: str) -> Tuple[str, Dict]:
    """Helper function to get response from EmotionalDecisionMaker"""
    handler = EmotionalDecisionMaker()
    return handler.process_query(query)
