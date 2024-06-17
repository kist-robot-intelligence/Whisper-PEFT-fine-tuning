# Whisper PEFT fine tuning을 위한 데이터셋을 생성하는 코드.

import librosa
from datasets import Dataset, DatasetDict
import glob
from sklearn.model_selection import train_test_split

# 생성된 데이터셋 저장 경로
#save_path = '[생성된 데이터셋을 저장할 경로]'
save_path = './generated_dataset'

# 오디오 파일 경로 자동 로드
#audio_file_paths = glob.glob('[파인튜닝에 사용할 wav 파일이 모여있는 폴더 경로]/*.wav')
audio_file_paths = glob.glob('./wav_files/*.wav')

# transcripts.txt 파일 파싱
transcripts_dict = {}
#with open('[파일명|transcripit 로 구성된 텍스트 파일]', 'r') as file:
with open('./filelist_vits.txt', 'r') as file:

    for line in file:
        audio_file_name, transcript = line.strip().split('|', 1)
        transcripts_dict[audio_file_name] = transcript

'''
[파일명|transcripit 로 구성된 텍스트 파일] 예제

slu-samples-5026-spk106.wav|나 안녕하지 않아
slu-samples-5027-spk110.wav|잘 쉬어
slu-samples-5028-spk114.wav|푹 쉬어
slu-samples-5029-spk118.wav|가서 쉬어
slu-samples-5030-spk122.wav|셔
slu-samples-5031-spk126.wav|쉬어

위와 같이 구성하면 됨

'''

# 오디오 파일명과 매칭되는 텍스트 및 오디오 데이터를 데이터셋에 추가
data = {'audio': [], 'transcript': []}
for path in audio_file_paths:
    audio_file_name = path.split('/')[-1]
    if audio_file_name in transcripts_dict:
        # 오디오 파일 로드
        audio, sampling_rate = librosa.load(path, sr=16000)  # 샘플링 레이트 16000으로 변경, librosa 기본은 22050이기 때문에 Whisper에서 사용하는 샘플링 레이트인 16000으로 변경해야 함.
        data['audio'].append({'path': audio_file_name, 'array': audio, 'sampling_rate': sampling_rate})
        data['transcript'].append(transcripts_dict[audio_file_name])
    else:
        print(f"Warning: No transcript found for {audio_file_name}")

# 데이터를 훈련 세트와 테스트 세트로 나눔
train_data, test_data = train_test_split(list(zip(data['audio'], data['transcript'])), test_size=0.1, random_state=42)

# 훈련 세트와 테스트 세트를 별도의 딕셔너리로 재구성
train_dataset = {'audio': [item[0] for item in train_data], 'transcript': [item[1] for item in train_data]}
test_dataset = {'audio': [item[0] for item in test_data], 'transcript': [item[1] for item in test_data]}

# 데이터셋 생성
audio_dataset = DatasetDict({
    'train': Dataset.from_dict({'audio': train_dataset['audio'], 'transcript': train_dataset['transcript']}),
    'test': Dataset.from_dict({'audio': test_dataset['audio'], 'transcript': test_dataset['transcript']})
})

# 데이터셋을 허깅페이스 서버에 저장
#audio_dataset.push_to_hub('martinGale/synthesized_corpus_ver_2')  # 허깅페이스 데이터셋 저장소에 push하는 코드

# 데이터셋을 로컬에 저장
audio_dataset.save_to_disk(save_path) # 로컬 주소에 데이터셋을 저장하는 코드

# 결과 확인
print(audio_dataset['train'][0])  # 첫 번째 항목의 오디오 데이터와 텍스트 출력

