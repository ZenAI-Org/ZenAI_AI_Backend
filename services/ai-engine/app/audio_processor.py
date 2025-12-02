import os 
from openai import OpenAI 
from fastapi import UploadFile, HTTPException 
import aiofiles 

class AudioProcessor: 
    def __init__(self):
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")) 
        self.supported_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
        
    async def save_upload_file(self, upload_file : UploadFile, destination: str): 
        # Saves uploaded file to disk temporarily 
        async with aiofiles.open(destination, 'wb') as out_file: 
            content = await upload_file.read() 
            await out_file.write(content) 
        
        def transcribe_audio(self,audio_file_path : str) -> str: 
            # Transcribe audio file using OpenAI's whisper model 
            try : 
                print(f"[INFO] Transcribing audio file: {audio_file_path}") 
                
                with open(audio_file_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model = "whisper-1", 
                        file = audio_file, 
                        response_format = "text" 
                    ) 
                print(f"[INFO] Transcription succesful, length : {len(transcript)}  characters")
                return transcript 
            except Exception as e : 
                print(f" [ERROR] Transcription failed : {e}") 
                raise HTTPException(status_code = 500, detail = f"Transcription failed: {str(e)}") 
        
        def cleanup_file(self , file_path : str): 
            # Remove temporary file from disk 
            try : 
                if os.path.exists(file_path): 
                    os.remove(file_path) 
                    print(f"[INFO] Removed temporary file: {file_path}")   
            except Exception as e :
                print(f"[WARNING] Failed to remove temporary file {file_path} : {e} ")