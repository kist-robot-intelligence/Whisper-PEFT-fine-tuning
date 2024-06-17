# Whisper PEFT 파인튜닝

## 소개

1. wav_files 디렉토리에 학습시킬 wav 파일들을 넣음.
2. dataset_generator.py를 실행시키면 wav_files 안의 음성 파일로부터 generated_dataset 폴더에 fine_tuning_PEFT.ipynb 파일에서 사용되는 데이터셋을 생성함.
3. fine_tuning_PEFT.ipynb를 실행시키면 파인튜닝이 진행됨.
4. 파인튜닝 완료 후 06171332와 같은 형태의 디렉토리가 만들어지고, 해당 폴더 안에 파인튜닝 된 adapter가 들어 있음.
