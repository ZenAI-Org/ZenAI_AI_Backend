import os 
from openai import OpenAI 
from fastapi import UploadFile, HTTPException 
import aiofiles 

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class AudioProcessor: 
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        self.client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        
        if self.google_key and genai:
            genai.configure(api_key=self.google_key)
            self.gemini_available = True
        else:
            self.gemini_available = False
            
        self.supported_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
        
    async def save_upload_file(self, upload_file: UploadFile, destination: str): 
        # Saves uploaded file to disk temporarily 
        async with aiofiles.open(destination, 'wb') as out_file: 
            content = await upload_file.read() 
            await out_file.write(content) 
        
    def transcribe_audio(self, audio_file_path: str, use_gemini: bool = True) -> str: 
        # Transcribe audio file using Gemini (Default) or OpenAI's whisper model
        
        # 1. Try Gemini if available (Primary)
        if (use_gemini or not self.client) and self.gemini_available:
            try:
                print(f"[INFO] Transcribing with Gemini 1.5 Pro: {audio_file_path}")
                model = genai.GenerativeModel("gemini-1.5-pro")
                
                # Upload the file to Gemini (File API)
                uploaded_file = genai.upload_file(audio_file_path)
                
                # Generate content (prompt + audio)
                response = model.generate_content(
                    ["Generate a specialized meeting transcript for this audio.", uploaded_file]
                )
                
                transcript = response.text
                print(f"[INFO] Gemini transcription successful, length: {len(transcript)} characters")
                return transcript
                
            except Exception as e:
                print(f"[ERROR] Gemini transcription failed: {e}")
                # Fallback to OpenAI if configured
                if not self.client:
                    raise HTTPException(status_code=500, detail=f"Gemini transcription failed and OpenAI not available: {str(e)}")
        
        # 2. Use OpenAI Whisper
        print(f"[INFO] Falling back to OpenAI Whisper")
        
        # 2. Use OpenAI Whisper
        if not self.client:
             raise HTTPException(status_code=500, detail="No Audio provider (OpenAI or Gemini) configured")
             
        try: 
            print(f"[INFO] Transcribing audio file (Whisper): {audio_file_path}") 
            
            with open(audio_file_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="text" 
                ) 
            print(f"[INFO] Transcription successful, length: {len(transcript)} characters")
            return transcript 
        except Exception as e: 
            print(f"[ERROR] Transcription failed: {e}") 
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") 
    
    def cleanup_file(self, file_path: str): 
        # Remove temporary file from disk 
        try: 
            if os.path.exists(file_path): 
                os.remove(file_path) 
                print(f"[INFO] Removed temporary file: {file_path}")   
        except Exception as e:
            print(f"[WARNING] Failed to remove temporary file {file_path}: {e}")