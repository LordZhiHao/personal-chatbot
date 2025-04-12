# Enhanced speech_to_text.py with better WAV file handling
import speech_recognition as sr
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import logging
import wave
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

class TranscriptionResponse(BaseModel):
    text: str

@router.post("/upload-audio", response_model=TranscriptionResponse)
async def upload_audio(file: UploadFile = File(...)):
    temp_filename = None
    converted_filename = None
    
    try:
        # Log file details
        logger.info(f"Received file: {file.filename}, content-type: {file.content_type}")
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_filename = temp_file.name
            content = await file.read()
            logger.info(f"File size: {len(content)} bytes")
            temp_file.write(content)
        
        # For WAV files, verify they can be opened with wave module
        if temp_filename.lower().endswith('.wav'):
            try:
                # Try to open with wave to verify format
                with wave.open(temp_filename, 'rb') as wave_file:
                    logger.info(f"WAV file details: channels={wave_file.getnchannels()}, "
                               f"width={wave_file.getsampwidth()}, "
                               f"rate={wave_file.getframerate()}, "
                               f"frames={wave_file.getnframes()}")
            except wave.Error as wave_err:
                logger.warning(f"Invalid WAV file: {wave_err}")
                raise HTTPException(status_code=400, detail=f"Invalid WAV file: {str(wave_err)}")
            
            # File seems valid, proceed with recognition
            audio_file_path = temp_filename
        else:
            # For non-WAV files, try to convert using FFmpeg if available
            logger.info("Received non-WAV file, trying to convert...")
            try:
                import subprocess
                
                # Create a new temp file for the converted WAV
                converted_filename = tempfile.mktemp(suffix='.wav')
                
                # Run FFmpeg to convert
                command = [
                    'ffmpeg',
                    '-i', temp_filename,
                    '-ar', '16000',  # Sample rate
                    '-ac', '1',      # Mono
                    '-f', 'wav',     # Format
                    converted_filename
                ]
                
                logger.info(f"Running conversion command: {' '.join(command)}")
                result = subprocess.run(command, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    raise Exception(f"Could not convert audio: {result.stderr}")
                
                audio_file_path = converted_filename
                logger.info(f"Successfully converted to WAV: {audio_file_path}")
                
            except Exception as e:
                logger.error(f"Conversion error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Unsupported audio format. Only WAV files are supported directly. Conversion failed: {str(e)}")
        
        try:
            # Initialize recognizer with adjustable parameters
            recognizer = sr.Recognizer()
            
            # Make recognition more sensitive
            recognizer.energy_threshold = 300  # Default is 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8
            
            logger.info(f"Processing file: {audio_file_path}")
            
            # Process audio with detailed error handling
            try:
                with sr.AudioFile(audio_file_path) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Record audio
                    audio_data = recognizer.record(source)
                    logger.info("Audio data recorded from file")
            except Exception as audio_error:
                logger.error(f"Error processing audio file: {str(audio_error)}")
                raise HTTPException(status_code=400, 
                                  detail=f"Could not process audio file. Please ensure it's a valid WAV file. Error: {str(audio_error)}")
            
            # Try Google's Speech Recognition
            try:
                logger.info("Sending to Google Speech Recognition...")
                text = recognizer.recognize_google(audio_data)
                logger.info(f"Google transcription result: {text}")
                
                return TranscriptionResponse(text=text)
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                raise HTTPException(status_code=400, detail="Could not understand audio. Please speak more clearly.")
            except sr.RequestError as req_error:
                logger.error(f"Google Speech Recognition request error: {str(req_error)}")
                raise HTTPException(status_code=503, 
                                  detail=f"Could not request results from Google Speech Recognition service. Check your internet connection.")
        
        finally:
            # Clean up the temporary files
            for filename in [temp_filename, converted_filename]:
                if filename and os.path.exists(filename):
                    os.remove(filename)
                    logger.info(f"Temporary file {filename} removed")
    
    except HTTPException:
        # Re-raise HTTP exceptions without modifying them
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")