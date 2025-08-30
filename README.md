# Word Window Classification

문장에서 위치에 해당하는 단어 찾기

위치를 의미하는지 판단할 떄 하나의 단어만 보고 판단하는 것이 아닌 주변 단어도 고려하여 판단함 (문맥을 고려)

## 실행

```
# git bash
git clone https://github.com/finale22/word-window-classifier.git
cd word-window-classifier
```
```
python -m venv myenv
source myenv/Scripts/activate
pip install -r requirements.txt
```
또는
```
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

---

bash 터미널에서 `python run.py --mode test` 입력하여 사용 가능

## 한계점

1. 임의의 몇 개의 문장을 생성하여 이를 데이터로 설정하고 10 에포크만 학습 $$\rightarrow$$ 성능 향상을 위해선 더 많은 데이터로 더 많이 학습해야 함

2. 출력은 입력 문장의 모든 단어에 대해 위치를 의미할 확률을 0 ~ 1 사이의 값(Sigmoid)으로 표현 -> 결국 확률 분포의 max는 단 하나의 단어

   그럼 위치를 의미하는 단어가 2개 이상인 경우에는? $$\rightarrow$$ 임계값을 기준으로 판단하면 의미가 있는가?
