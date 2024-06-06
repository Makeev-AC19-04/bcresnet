from bcresnet import BCResNets
import torch
import torchaudio
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import pyaudio
import os
import time
from threading import Thread
import psutil

from tqdm import tqdm
from utils import Padding, Preprocess, SpeechCommand, SplitDataset, LogMel, spec_augment

from constants import SAMPLE_RATE, THRESHOLD, NUM_OF_CHUNKS, RECORD_SECONDS, KEYWORD, TAU

gpu = 1
device = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")

process_times = []
process_memo = []

def process_sample(sample):
    sample = torch.clamp(sample, -1.0, 1.0)
    SR = 16000
    hop_length=160
    win_length=480
    n_fft=512
    n_mels=40
    feature = LogMel(
                device,
                sample_rate=SR,
                hop_length=hop_length,
                win_length=win_length,
                n_fft=n_fft,
                n_mels=n_mels,
            )
    sample = feature(sample)  
    sample = spec_augment(sample)
    return sample


def predict_sample(sample, launch_time, model, i):
    global process_times, process_memo
    start_time = time.time()
    memory_before = psutil.virtual_memory().used

    # Применяем шумоподавление
    nr_sample = torch.tensor(nr.reduce_noise(y=sample, sr=SAMPLE_RATE, stationary=True))

    # Process the audio tensor with feature extraction and classification
    sample_processed = process_sample(nr_sample)
    sample_processed = sample_processed.to(device)

    predictions = model(sample_processed.unsqueeze(0))
    
    end_time = time.time()
    memory_after = psutil.virtual_memory().used
    time_used = end_time-start_time
    memo_used = (memory_after-memory_before)/1024
    
    print(f"Iteration: {i}, time to process: {time_used:.4f}, memory: {memo_used} Kb, time from the start:{end_time-launch_time}")
    print(predictions)

    if i >= NUM_OF_CHUNKS:
        if predictions[0][1].item() > THRESHOLD:
            print("Keyword")
        else:
            print("Filler")
    process_times.append(time_used)
    process_memo.append(memo_used)    

def stream_process():
    global process_times, process_memo
    model = BCResNets(int(TAU * 8)).to(device)
    model.load_state_dict(torch.load(F"models/{KEYWORD}_model.pth"))
    model.eval()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = int(SAMPLE_RATE * RECORD_SECONDS/NUM_OF_CHUNKS)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)

    sample_window = int(SAMPLE_RATE * RECORD_SECONDS) # How much samples in recorded audio
    chunks_num = int(sample_window/CHUNK) # How much chunks in recorder audio

    sample = torch.zeros(1, sample_window)
    i = 0
    launch_time = time.time()
    try:
        while True:
            i+=1
            # Запускаем поток с обработкой сигнала, чтобы параллельно записывать следующий чанк
            thread = Thread(target=predict_sample, args = (sample, launch_time, model, i,))
            thread.start()

            # Считываем чанк аудиосигнала с микрофона
            data = stream.read(CHUNK) 
            
            # Преобразуем его в тензор
            audio_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.int16) / 32767.0) 

            # Передвигаем значения сэмпла от второго до последнего чанка в диапазон от первого до предпоследнего чанка
            sample[0, 0:(chunks_num - 1)*CHUNK] = sample[0, CHUNK:chunks_num*CHUNK].clone() 
            sample[0, (chunks_num - 1)*CHUNK:chunks_num*CHUNK] = audio_tensor

            # Перед тем, как обработать следующий чанк, ждем завершения потока обработки предыдущего 
            thread.join()
            
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

    plt.plot(process_memo[1:])
    plt.xlabel('Iteration')
    plt.ylabel('Memory for process, Kb')
    plt.savefig('reports/process_memo.png')
    plt.clf()

    plt.plot(process_times[1:])
    plt.axhline(y=RECORD_SECONDS/NUM_OF_CHUNKS, color='r', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Time for process, seconds')
    plt.savefig('reports/process_times.png')

if __name__ == "__main__":
    stream_process()