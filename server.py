import os
import uuid
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pipeline import full_pipeline
import speech_recognition as sr
import io

app = FastAPI()

@app.get("/")
async def get():
    with open("client.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        current_lang = "fra_Latn"
        
        while True:
            # Receive message
            message = await websocket.receive()
            
            if "text" in message:
                try:
                    import json
                    text_data = json.loads(message["text"])
                    if "language" in text_data:
                        current_lang = text_data["language"]
                        print(f"Language updated to: {current_lang}")
                        await websocket.send_json({"status": "info", "message": f"Language set to {current_lang}"})
                    continue
                except Exception as e:
                    print(f"Error parsing text message: {e}")
                    continue

            if "bytes" in message:
                data = message["bytes"]
                print(f"Received audio tokens: {len(data)} bytes")
                
                if len(data) < 2000:
                    print("Ignoring small audio packet")
                    continue
            
                # Create unique filenames for this request
                session_id = str(uuid.uuid4())
                input_filename = f"temp_input_{session_id}.wav"
                output_filename = f"temp_output_{session_id}.wav"
                
                try:
                    # Save received audio to file
                    # Assuming the client sends a WAV file directly
                    with open(input_filename, "wb") as f:
                        f.write(data)
                    
                    # Run the pipeline
                    print(f"Running pipeline with lang={current_lang}...")
                    refined_text, generated_audio_path = await asyncio.to_thread(
                        full_pipeline, 
                        audio_input=input_filename, 
                        target_lang=current_lang,
                        output_audio_path=output_filename
                    )
                    
                    print(f"Pipeline finished. Text: {refined_text}")
                    
                    # Check if output file exists
                    if os.path.exists(generated_audio_path):
                        with open(generated_audio_path, "rb") as audio_file:
                            audio_data = audio_file.read()
                            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        
                        # Send response back to client
                        await websocket.send_json({
                            "text": refined_text,
                            "audio": audio_base64,
                            "status": "success"
                        })
                    else:
                         await websocket.send_json({
                            "status": "error",
                            "message": "Output audio generation failed"
                        })

                except Exception as e:
                    print(f"Error processing request: {e}")
                    await websocket.send_json({
                        "status": "error",
                        "message": str(e)
                    })
                
                finally:
                    # Cleanup temp files
                    if os.path.exists(input_filename):
                        os.remove(input_filename)
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
                    
    except WebSocketDisconnect:
        print("Client disconnected")
