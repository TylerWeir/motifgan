"""
Usage: `python augment.py <data_path>`
Takes in command line argument for directory containing the data in the format:
data
├── negative
└── positive
The parent directory doesn't have to be named 'data' but the children directories must be called "negative" and "positive". 
Inside of "positive" and "negative" should be only audio files
AUGMENTATION FUNCTIONS:
Should accept two positional arguments, `clip` and `outpath_stub`, as well as at least one keyword argument, `duplicate_original`.
Augmentation functions should save both a normalized copy of the clip and an augmented version. See distort() for a example.
"""

import os
import sys
import random
from random import randrange
from math import ceil

import numpy as np
import soundfile as sf
import pyrubberband
from pydub import AudioSegment
from pydub.generators import WhiteNoise


### Distortion

def distort(clip : AudioSegment, outpath_stub, duplicate_original=True):
    """
    ## Distort clip by a randomized amount and save.
    Also saves a copy without added distortion
    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./distorted_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './distorted_clips/myclip1.mp3_copy.mp3'\ 
    './distorted_clips/myclip1.mp3_distorted_0.mp3'\ 
    './distorted_clips/myclip1.mp3_distorted_1.mp3'
    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    # save a copy
    if duplicate_original:
        clip.export(f"{outpath_stub}_copy.mp3", format="mp3")
    
    # augment
    gain = random.uniform(6,12)
    clip = clip + gain # add 8 to 16 db gain

    # save the augmented version
    clip.export(f"{outpath_stub}_distorted_0.mp3", format="mp3")

    clip = clip + gain
    clip.export(f"{outpath_stub}_distorted_1.mp3", format="mp3")



### Noise

def addnoise(clip : AudioSegment, outpath_stub, duplicate_original=True):
    """
    ## Combine clip with a random amount of white noise and save.
    Also saves a copy without added noise
    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./noisy_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './noisy_clips/myclip1.mp3_copy.mp3'\ 
    './noisy_clips/myclip1.mp3_noisy_0.mp3'\ 
    './noisy_clips/myclip1.mp3_noisy_1.mp3'
    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    if(duplicate_original):
        clip.export(f"{outpath_stub}_copy.mp3", format="mp3")

    noise = WhiteNoise().to_audio_segment(duration=len(clip))
    noise_level = random.uniform(-46, -32)

    combined = noise.overlay(clip, gain_during_overlay=noise_level)
    combined.export(f"{outpath_stub}_noisy_0.mp3", format="mp3")

    noise_level = random.uniform(-32, -24)

    combined = noise.overlay(clip, gain_during_overlay=noise_level)
    combined.export(f"{outpath_stub}_noisy_1.mp3", format="mp3")
    

### Timeshifting 

def timeshift(clip : AudioSegment, outpath_stub, min_shift=100, max_shift=800, 
            num_times=1, duplicate_original=True):
    """
    # Add a random amount of silence around the clip and save.
    Also saves a copy without timeshifing
    
    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.
        Ex: with outpath_stub=`./timeshift_clips/myclip1.mp3` output may look like:
        `./timeshift_clips/myclip1_copy.mp3`
        `./timeshift_clips/myclip1_timeshift_0.mp3`
        `./timeshift_clips/myclip1_timeshift_1.mp3`
    
    - `min_shift` -- The minium number of milliseconds the timeshift may be. Default is 250 ms.
    - `max_shift` -- The maximum number of milliseconds the timeshift may be. Default is 1000 ms. 
    - `num_times` -- The number of times to copy the original sample and add a random timeshift. Default is 3
        
    - `dulplicate_original -- Whether or not to include a copy of the original in the output.
                            defaults to true.
    """

    if duplicate_original:
        clip.export(f'{outpath_stub}_copy.mp3', format="mp3")

    for i in range(num_times):
        beginning_silence = AudioSegment.silent(duration=randrange(min_shift, max_shift))
        ending_silence = AudioSegment.silent(duration=randrange(min_shift, max_shift))
        output = beginning_silence + clip + ending_silence
        output.export(f'{outpath_stub}_timeshift_{i}.mp3', format="mp3")





#######################
### Time Stretching ###
#######################

# this was taken from https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa also using https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
def audiosegment_to_numpy_array(audiosegment):
    """ extracts and returns the numpy array of samples from a pydub AudioSegment """

    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds[:1]]

    print(len(samples[0]))

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    print(np.shape(fp_arr))

    return fp_arr

def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    i.e. if chunk_length is 500 then you'll get a list of 500ms long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length:(i + 1) * chunk_length] for i in range(number_of_chunks)]

def timestretch(clip : AudioSegment, outpath_stub, duplicate_original=True):
    """## Timestretch segments of clip at varying speeds and save.
    Also saves a copy without time-stretching.
    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./stretched_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './stretched_clips/myclip1.mp3_copy.wav'\ 
    './stretched_clips/myclip1.mp3_stretched_0.wav'\ 
    './stretched_clips/myclip1.mp3_stretched_1.wav'
    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    if duplicate_original:
        clip.export(f"{outpath_stub}_copy.wav", format="wav")

    sample_rate = clip.frame_rate
    clip_length = clip.duration_seconds
    
    # split up audio into chunks
    num_chunks = randrange(3,8) # [3,8)
    chunk_length = ceil(1000 * clip_length / num_chunks)
    chunks = make_chunks(clip, chunk_length)

    combined = np.array([])
    for chunk in chunks:
        # stretch each chunk by a random amount and append it
        rate = random.uniform(0.7, 1.3)
        audio_array = audiosegment_to_numpy_array(chunk)
        to_append = pyrubberband.pyrb.time_stretch(audio_array, sample_rate, rate)
        combined = np.append(combined, to_append)
    
    # save as wav, because soundfile doesnt know how to save as mp3
    sf.write(f"{outpath_stub}_stretched.wav", combined, sample_rate)


### Pitch Shifting

def pitch_shift(clip : AudioSegment, outpath_stub, duplicate_original=True, num_times=3):
    """
    # Pitch shifts the clip and saves.
    Also saves a copy without pitch shifting
    
    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.
        Ex: with outpath_stub=`./timeshift_clips/myclip1.mp3` output may look like:
        `./timeshift_clips/myclip1_copy.mp3`
        `./timeshift_clips/myclip1_pitchshift_0.mp3`
        `./timeshift_clips/myclip1_pitchshift_1.mp3`
    
    - `min_shift` -- The minium number of milliseconds the timeshift may be. Default is 250 ms.
    - `max_shift` -- The maximum number of milliseconds the timeshift may be. Default is 1000 ms. 
    - `num_times` -- The number of times to copy the original sample and add a random timeshift. Default is 3
        
    - `dulplicate_original -- Whether or not to include a copy of the original in the output.
                            defaults to true.
    """
    if duplicate_original:
        clip.export(f'{outpath_stub}_copy.mp3', format="mp3")
    
    sr = clip.frame_rate
    print("sr=", sr)
    audio_data = audiosegment_to_numpy_array(clip)
    audio_data_copy = audiosegment_to_numpy_array(clip)

    for i in range(num_times):
        audio_data = pyrubberband.pitch_shift(audio_data, sr, 6)
        sf.write(f"{outpath_stub[:-4]}_pitchshifted_up_{i}.wav", audio_data, sr)


    for i in range(num_times):
        audio_data_copy = pyrubberband.pitch_shift(audio_data_copy, sr, -6)
        sf.write(f"{outpath_stub[:-4]}_pitchshifted_down_{i}.wav", audio_data_copy, sr)



def augment(data_dir):
    """
    augments the input data in stages
    returns the directory of the final stage
    """

    stage_num = 0
    for pos_or_neg in ["positive"]:    
        for index, augmentation in enumerate([pitch_shift]):
            stage_num = index + 1

            indir = f"{data_dir}/{pos_or_neg}" if index == 0 else f"./data/augmented_stage_{index}/{pos_or_neg}"
            outdir = f"./data/augmented_stage_{stage_num}/{pos_or_neg}"
            if not os.path.isdir(f"./data/augmented_stage_{stage_num}"):
                os.mkdir(f"./data/augmented_stage_{stage_num}")
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            
            counter = 0
            for item in os.listdir(indir):
                try:
                    clip = AudioSegment.from_file(f"{indir}/{item}")

                    # apply augmentation
                    print(f"Augmenting {item} with {augmentation.__name__}")
                    augmentation(clip, f"{outdir}/{item}")

                    # limit to one base example for testing
                    # if index == 0:
                    #     break
                except Exception as e:
                    print(e)

    
    return f"./data/augmented_stage_{stage_num}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        augment(sys.argv[1])
    else:
        print("Usage: `python augment.py <data_path>`")