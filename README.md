# Real-Time Transcription and TTS

Real-time speech transcription, natural language processing, and text-to-speech functionalities using Whisper, OpenAI, and ElevenLabs APIs.

## Technical Overview

### Dependencies

- **Whisper**: OpenAI's Whisper model is used for automatic speech recognition (ASR), enabling the transcription of spoken language into text.
    - **Model Loading**: The model is loaded using `whisper.load_model("base")`, which loads a pre-trained model suitable for general transcription tasks.
    - **Transcription**: The `model.transcribe()` method is employed to convert audio data into text. The audio is processed from raw data to a floating-point format suitable for the model.

- **OpenAI**: The OpenAI API provides advanced language models for generating text-based responses based on input prompts.
    - **API Client**: Initialized with `OpenAI(api_key=openai_api_key)`, where `api_key` is securely loaded from environment variables.
    - **Text Generation**: Uses `openai_client.chat.completions.create()` with model `gpt-3.5-turbo` to generate contextual responses. This involves sending a conversation history to the model to produce a coherent reply.

- **ElevenLabs**: ElevenLabs API is used for text-to-speech (TTS) functionality, converting generated text responses into audible speech.
    - **API Client**: Initialized with `ElevenLabs(api_key=elevenlabs_api_key)`, where `api_key` is securely loaded from environment variables.
    - **Audio Generation**: The `generate()` method converts text into speech, and `play()` is used to output the audio. The `voice` parameter specifies the synthetic voice used.

- **Sounddevice**: Handles real-time audio streaming and recording.
    - **Audio Streaming**: Uses `sd.InputStream` to capture audio data in real-time. The audio callback function processes incoming data and converts it to the format required by Whisper.


### Implementation Details

1. **Initialization**:
   - API clients for OpenAI and ElevenLabs are instantiated using API keys loaded from environment variables.
   - The Whisper model is loaded for transcription tasks.

2. **Real-Time Transcription**:
   - **`start_transcription()`**: Activates real-time audio capture. The `callback()` function processes incoming audio data, performs transcription, and triggers further actions based on the transcribed text.
   - **Error Handling**: Includes basic error handling for transcription and streaming errors.

3. **Response Generation**:
   - **`generate_ai_response(transcript)`**: Takes the transcribed text, appends it to the conversation history, and generates a response using the OpenAI API. The response is then converted to speech.
   - **Error Handling**: Manages potential errors in response generation.

4. **Text-to-Speech (TTS)**:
   - **`generate_audio(text)`**: Converts the AI-generated response into audible speech using ElevenLabs’ TTS capabilities. The speech is played back to the user.

5. **Testing and Execution**:
   - The assistant starts with a greeting message and begins transcription.
   - Transcription is stopped after a specified duration (e.g., 10 seconds) for testing purposes.

### Installation and Setup

1. **Environment Setup**:
   - Ensure Python 3.7+ is installed.
   - Install required libraries:
     ```bash
     pip install whisper elevenlabs openai sounddevice python-dotenv numpy
     ```

2. **Configuration**:
   - Create a `.env` file in the project root with the following content:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ELEVENLABS_API_KEY=your_elevenlabs_api_key
     ```

3. **Running the Application**:
   - Execute the script using Python:
     ```bash
     python app.py
     ```
