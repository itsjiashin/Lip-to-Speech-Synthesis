# Lip-to-Speech Synthesis

The lip-to-speech synthesis technology for non-English languages remains relatively unexplored within the current scientific literature. This project will build upon previous works to produce a model that is able to synthesize speech based on lip motions of a speaker for multiple languages, not just English.  By doing so, cross-language communication can be improved upon.

## Acknowledgements
This repository builds upon the PyTorch implementation of the following paper:
> **Lip to Speech Synthesis with Visual Context Attentional GAN**<br>
> Minsu Kim, Joanna Hong, and Yong Man Ro<br>
> \[[Paper](https://proceedings.neurips.cc/paper/2021/file/16437d40c29a1a7b1e78143c9c38f289-Paper.pdf)\] [[Code](https://github.com/ms-dot-k/Visual-Context-Attentional-GAN)\]

## Requirements
In order to run the program, please ensure that the necessary Python libraries stated below are installed. Calling pip install followed by the python library is sufficient enough to download the libraries. The necessary Python libraries are:
- python (version 3.7 or above)
- pytorch (version 1.6 to 1.8)
- torchvision
- torchaudio
- ffmpeg
- av
- tensorboard
- scikit-image (version 0.17.0 or above)
- opencv-python (version 3.4 or above)
- pillow
- librosa
- pystoi
- pesq
- scipy (version 1.11.3 or above)
- mediapipe
- moviepy (version 1.0.3 or above)
- numpy
- soundfile

## Preprocessing the dataset
1. Extracting Audio <br>
`preprocess_audio_only.py` extracts audio from the video.<br>
```shell
python preprocess_audio_only.py \
--Grid_dir "Data directory of audio-visual corpus"
--output_dir "Output directory of audio of the audio-visual Corpus"
```
2.  Run crop_video_processing.py to obtain videos that are cropped to the speaker’s faces. This code only works with single speaker videos, therefore corrupted or multi-speaker videos will be deleted. Within the code, the directories should be changed to where the audio-visual corpus is extracted to.

3.  Run remove_not_25.py to remove all the cropped videos obtained from step 2 that are not 25 FPS. Within the code, please change the directory to the directory where the cropped videos from step 2 are stored. The code will generate a text file telling you which file is not 25 fps. Delete all the videos within the text file as it cannot be used by the model.

4. At this point, there should be audio files corresponding to cropped muted videos that are 25 FPS. Now, cut the remaining videos and audios to exactly 3 seconds long. Only the first 3 seconds of every audio and video clip is required. This can be done manually using an application called XMedia Recode.

5. Make a folder for every video left. Inside that folder, there should be two subfolders titled video and audio respectively, where the video subfolder houses the 3 second long video clip, and the audio subfolder houses the 3 second audio clip for the video.

6. Preprocessing of the dataset is done. In order to load the data for training, validation and testing, modify them in the training.txt, validation.txt and test.txt under the data folder.

## Training the model
To train the model, run the following commands:
```shell
python train.py \
--grid 'Directory that contains the preprocessed data' \
--checkpoint_dir 'Directory where the checkpoint should be saved at' \
--batch_size 88 \
--epochs 500 \
--eval_step 720 \
--lr 0.0001 \ 
--dataparallel \
--gpu 0,1,2,3
```
Descriptions of training parameters are as follows:
- `--grid`: Dataset location 
- `--checkpoint_dir`: Directory for saving checkpoints
- `--batch_size`: batch size 
- `--epochs`: number of epochs
- `--eval_step`: steps for performing evaluation
- `--lr`: learning rate
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training

## Testing the model 
To test the model, run the following commands:
```shell
# Dataparallel test example for multi-speaker setting in GRID
python test.py \
--grid 'Directory that contains the preprocessed data' \
--batch_size 100 \
--save_mel \
--save_wav \
--dataparallel \
--gpu 0,1
```

Descriptions of training parameters are as follows:
- `--grid`: Dataset location 
- `--batch_size`: batch size 
- `--save_mel`: whether to save the 'mel_spectrogram' and 'spectrogram' in `.npz` format
- `--save_wav`: whether to save the 'waveform' in `.wav` format
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training






