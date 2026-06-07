# ADC Score Predictor

`model.ipynb`와 동일한 CDR3 특성 및 학습 결과를 사용하는 웹 애플리케이션입니다.
Vercel에서는 정적 프론트엔드와 Python Serverless Function으로 동작합니다.

## Vercel 배포

GitHub 저장소를 Vercel에 연결할 때 프로젝트 설정의 **Root Directory**를
`simulation`으로 지정합니다. Framework Preset은 `Other`로 두면 됩니다.
별도의 Build Command와 환경 변수는 필요하지 않습니다.

Vercel CLI를 사용할 경우 프로젝트 루트에서 다음과 같이 실행합니다.

```bash
cd simulation
npx vercel
npx vercel --prod
```

배포 구조:

- `public/`: 정적 웹페이지
- `api/predict.py`: 메타데이터 조회 및 ADC score 예측 API
- `lib/model_artifact.json`: 로컬에서 학습해 내보낸 모델 파라미터
- `lib/adc_runtime.py`: ANARCI 기반 CDR3 추출 및 경량 추론

Vercel 함수는 요청마다 모델을 재학습하지 않습니다. 새 데이터로
`model.ipynb`의 학습 결과를 변경했다면 아래 명령으로 모델 아티팩트를 갱신한 후
다시 배포합니다.

```bash
venv/bin/python simulation/export_model.py
```

## 로컬 실행

프로젝트 루트에서:

```bash
venv/bin/python simulation/app.py
```

브라우저에서 `http://127.0.0.1:8000`에 접속합니다.

다른 기기에서도 접속하도록 열려면:

```bash
venv/bin/python simulation/app.py --host 0.0.0.0 --port 8000
```

별도 개발 환경에서는 먼저 의존성을 설치합니다.

```bash
python -m pip install -r simulation/requirements-dev.txt
```

서버 시작 시 `dataset/approved.CSV`, `phase3.CSV`, `phase2.CSV`,
`phase1.CSV`를 읽어 모델을 한 번 학습합니다. 입력된 heavy/light chain에서
HCDR3와 LCDR3를 추출해 ADC score를 1~10 범위로 반환합니다.
