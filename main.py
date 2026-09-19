import asyncio
import sys
import os
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import pygame
import queue
from scipy.io import wavfile
import wave
from shazamio import Shazam
from constants import *

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Album Art")

image_url = ""
wav_buffer = io.BytesIO()
chunk = MIN_CHUNK_DURATION
scaled = None
samples_per_chunk = SAMPLE_RATE * chunk


# Thread-safe queue to pass audio from the sounddevice thread to asyncio
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This callback is called for every audio block by the sounddevice thread."""
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the audio data into our thread-safe queue
    audio_queue.put(indata.copy())

async def listen():

    global wav_buffer,chunk, samples_per_chunk
    """Background asyncio task that pulls audio from the queue and processes it."""
    print("Microphone recording started...")
    
    # Store audio segments here until we hit CHUNK_DURATION
    recorded_chunks = []
    current_samples = 0

    print(f"Record a chunk ({chunk}) of {samples_per_chunk} samples")

    # Start the non-blocking input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            # Check the queue without blocking the event loop
            try:
                # Use block=False so we don't stall the asyncio loop if queue is empty
                data = audio_queue.get_nowait()
                recorded_chunks.append(data)
                current_samples += len(data)

                # Once we have 5 seconds of audio, process or save it
                if current_samples >= samples_per_chunk:
                    full_audio = np.concatenate(recorded_chunks, axis=0)
                    print(f"Recorded chunk: {len(full_audio)} samples (~{chunk}s)")
                    
                    # TODO: Send full_audio to an API, save to disk, or analyze frequencies here!
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, full_audio, SAMPLE_RATE, format="WAV", subtype = "PCM_16")

                    wav_buffer.seek(0)
                                    
                    print(f"wav_buffer {len(wav_buffer.getvalue())}")
                    await lookup_song()

                    # Reset buffer for the next chunk
                    recorded_chunks = []
                    current_samples = 0
                    
            except queue.Empty:
                # No new audio data available yet; yield control to the loop
                await asyncio.sleep(0.01)


async def lookup_song():
    global image_url
    global wav_buffer, chunk, samples_per_chunk

    if len(wav_buffer.getvalue()) == 0:
        return

    shazam = Shazam()

    result = None
    try: 
        result = await shazam.recognize(wav_buffer.getvalue())
    except ClientResponseError as e:
        print(f"HTTP Error Status Code: {e.status}")
        print(f"Message: {e.message}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Parse results safely
    if result and 'track' in result:
        track = result['track']
        title = track.get('title', 'Unknown Title')
        artist = track.get('subtitle', 'Unknown Artist')
        album = track.get('sections', [{}])[0].get('metadata', [{}])[0].get('text', 'Unknown Album')
        images = track.get("images")
        image_url = images.get("coverarthq", images.get("coverart"))
        
        print("\n🎵 Song Identified! 🎵")
        print(f"Title:  {title}")
        print(f"Artist: {artist}")
        chunk = MIN_CHUNK_DURATION
        samples_per_chunk = SAMPLE_RATE * chunk
    else:
        if chunk < MAX_CHUNK_DURATION:
            chunk += CHUNK_INCREMENT
            samples_per_chunk = SAMPLE_RATE * chunk
        print("\n❌ Could not identify the song. Try a clearer sample or extending the duration.")



async def main():
    global image_url
    global scaled
    asyncio.create_task(listen())

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()

        # update state
        screen.fill("black")
        if len(image_url) > 0:
            response = requests.get(image_url)
            image_file = io.BytesIO(response.content)
            raw_image = pygame.image.load(image_file)
            img = raw_image.convert_alpha()
            scaled = pygame.transform.scale(img, (IMAGE_SIZE, IMAGE_SIZE))

            image_url = ""

        if scaled: 
            screen.blit(scaled, ((SCREEN_WIDTH-IMAGE_SIZE)/2.0, (SCREEN_HEIGHT-IMAGE_SIZE)/2.0))

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":

    try:
        asyncio.get_running_loop()
        asyncio.create_task(main())
    except RuntimeError:
        asyncio.run(main())
