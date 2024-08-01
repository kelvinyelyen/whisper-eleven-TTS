import whisper
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import numpy as np
import sounddevice as sd
import io
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Define the AI_Assistant class
class AI_Assistant:
    def __init__(self):
        # Initialize API keys from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        self.openai_client = OpenAI(api_key=openai_api_key)

        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

        # Load Whisper model
        self.model = whisper.load_model("base")

        # Set initial values
        self.full_transcript = [
            {"role": "system", "content": "You are a receptionist at a school clinic. Be resourceful and efficient."}
        ]

        self.stream_active = False

    # Start real-time transcription with Whisper
    def start_transcription(self):
        self.stream_active = True
        
        def callback(indata, frames, time, status):
            if status:
                print(status, flush=True)

            # Convert the incoming audio data to a NumPy array and then to float32
            audio = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0  # Convert to float

            # Perform transcription
            try:
                result = self.model.transcribe(audio, fp16=False)
                transcript = result['text']
                self.on_data(transcript)
            except Exception as e:
                print(f"Error in transcription: {e}")

        # List available devices for debugging
        print("Available audio devices:")
        print(sd.query_devices())

        # Start streaming
        try:
            device_id = 1  # Use the appropriate device ID
            with sd.InputStream(device=device_id, callback=callback, channels=1, samplerate=16000):
                while self.stream_active:
                    sd.sleep(1000)
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop_transcription(self):
        self.stream_active = False

    # Callback for receiving transcript data
    def on_data(self, transcript):
        if not transcript:
            return

        self.generate_ai_response(transcript)

    # Pass real-time transcript to OpenAI
    def generate_ai_response(self, transcript):
        self.stop_transcription()
        self.full_transcript.append({"role": "user", "content": transcript})
        print(f"\nPatient: {transcript}")

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.full_transcript
            )
            ai_response = response.choices[0].message.content
            self.generate_audio(ai_response)
        except Exception as e:
            print(f"Error generating AI response: {e}")

        self.start_transcription()
        print(f"\nReal-time transcription:")

    # Generate audio with ElevenLabs
    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        try:
            # Generate and play audio from text
            audio = self.elevenlabs_client.generate(
                text=text,
                voice="Alice",
                model="eleven_multilingual_v2"
            )
            play(audio)
        except Exception as e:
            print(f"Error generating audio: {e}")

# Initial greeting and start the assistant
greeting = "Thank you for calling Academic City clinic. My name is Britney, if you have any questions do not hesitate to come by the clinic for a checkup we are available 24 hours of the day."
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)

# Start Transcription
ai_assistant.start_transcription()

# Stop Transcription after some time for testing (e.g., after 10 seconds)
import time
time.sleep(10)
ai_assistant.stop_transcription()
