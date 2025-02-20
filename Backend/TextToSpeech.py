import pygame
import edge_tts
import os
import json
import time
import asyncio
from dotenv import dotenv_values
# Add this at the start of the TTSEngine.__init__ method
os.makedirs("Data", exist_ok=True)

env_vars = dotenv_values(".env")
AssistantVoice = env_vars.get("AssistantVoice")

class TTSEngine:
    def __init__(self):
        self.audio_file = r"Data\speech.mp3"
        pygame.mixer.init()
        
        # Load emotion configurations
        try:
            with open(r"Config/emotions_config.json", "r") as f:
                self.emotion_config = json.load(f)
        except FileNotFoundError:
            self.emotion_config = {
                "voice_modulation": {
                    "happy": {"pitch": 1.5, "speed": 1.5},
                    "sad": {"pitch": 1.1, "speed": 1.2},
                    "angry": {"pitch": 1.6, "speed": 1.7},
                    "neutral": {"pitch": 1.3, "speed": 1.3}
                }
            }

    async def generate_audio(self, text: str, emotion_params: dict = None) -> None:
        """Generate audio file for the given text with emotion parameters"""
        try:
            # Clean up previous audio file
            if os.path.exists(self.audio_file):
                try:
                    os.remove(self.audio_file)
                    await asyncio.sleep(0.1)  # Give OS time to release file
                except PermissionError:
                    self.audio_file = f"Data\speech_{int(time.time())}.mp3"

            # Configure voice parameters
            voice_config = self.get_voice_config(emotion_params)
            
            # Generate audio
            communicate = edge_tts.Communicate(
                text, 
                AssistantVoice,
                pitch=voice_config['pitch'],
                rate=voice_config['rate'],
                volume='+0%'
            )
            
            await communicate.save(self.audio_file)
            await asyncio.sleep(0.1)  # Ensure file is written
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            raise

    def get_voice_config(self, emotion_params: dict) -> dict:
        """Get voice configuration based on emotion"""
        if not emotion_params:
            return {'pitch': '+0Hz', 'rate': '+0%'}

        emotion = emotion_params.get('emotion', 'neutral')
        confidence = emotion_params.get('confidence', 1.0)

        # Map emotions to basic categories
        emotion_mapping = {
            'admiration': 'happy', 'amusement': 'happy', 'joy': 'happy',
            'sadness': 'sad', 'disappointment': 'sad', 'grief': 'sad',
            'anger': 'angry', 'annoyance': 'angry', 'disgust': 'angry',
            'fear': 'worried', 'nervousness': 'worried',
            'excitement': 'excited', 'surprise': 'excited',
            'neutral': 'neutral'
        }

        base_emotion = emotion_mapping.get(emotion, 'neutral')
        voice_params = self.emotion_config['voice_modulation'].get(base_emotion, {})
        
        pitch_mod = voice_params.get('pitch', 1.2)
        rate_mod = voice_params.get('speed', 1.2)
        
        return {
            'pitch': f"{int((pitch_mod - 1.2) * 100 * confidence):+d}Hz",
            'rate': f"{int((rate_mod - 1.2) * 100 * confidence):+d}%"
        }

    def play_audio(self) -> None:
        """Play the generated audio file"""
        try:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Playback error: {e}")
            raise

def TextToSpeech(text: str, emotion_params: dict = None) -> bool:
    """Main TTS function that handles the complete TTS process"""
    engine = TTSEngine()
    
    try:
        # Split text into manageable chunks (sentences)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            # Generate and play audio for each sentence
            asyncio.run(engine.generate_audio(sentence, emotion_params))
            engine.play_audio()
            time.sleep(0.1)  # Brief pause between sentences
            
        return True
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return False
        
    finally:
        pygame.mixer.quit()
        pygame.mixer.init()  # Reinitialize for next use

if __name__ == "__main__":
    test_text = "This is a test of emotional text to speech. It should read everything completely."
    test_emotions = {'emotion': 'joy', 'confidence': 0.9}
    TextToSpeech(test_text, test_emotions)
