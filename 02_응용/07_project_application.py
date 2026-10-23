"""07~08 응용: 별도 모듈 없이 이 파일 하나로 완성합니다."""
# 0. 패키지 import
## 0-1. 경로 관련
from pathlib import Path
## 0-2. Data 관련
import pandas as pd
## 0-3. Data 전처리 및 modeling
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.

# 1. 기본 작업 경로 설정 및 데이터 import
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
data = pd.read_csv(HERE / "data" / "requests.csv")

# 2. 데이터 확인
data.info()
print(data.describe(include='all'))

# 3. 데이터 train_test_split
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.

# 4. 데이터 preprocessing
# 제공: 숫자 2열은 중앙값 대체 후 표준화합니다.
## 4-1. 수치형 변수처리 pipeline(결측 처리(중앙값) 및 StandardScaler 처리 준비)
numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

## 4-2. 범주형 변수처리 pipeline(결측 처리(최빈값) 및 OnehotEncoder 처리 준비)
# 제공: 문자 2열은 최빈값 대체 후 0/1 열로 바꿉니다.
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

### 4-3. 각 변수 전처리 pipeline 적용
preprocessor = ColumnTransformer([
    ("num", numeric, ["input_chars", "allowed_tools"]),
    ("cat", categorical, ["task_type", "priority"]),
])
features = ["input_chars", "allowed_tools", "task_type", "priority"]
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
print(train.head(5))
print(valid.head(5))
pipeline.fit(train[features], train["failed"])
# TODO A07-1: Pipeline으로 valid의 입력 4열을 예측하세요.
prediction = pipeline.predict(valid[features])

# 5. 결과 제시
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction
result.to_csv(OUTPUTS / "valid_predictions.csv", index=False, encoding="utf-8-sig")
# TODO A07-2: 실제값과 예측값이 같은 비율을 구하세요.
accuracy = (result["failed"] == result["predicted"]).mean()
found = ((result["failed"] == 1) & (result["predicted"] == 1)).sum()
print("valid 정확도:", round(accuracy, 2))
print("실패 발견:", int(found), "/", int((result["failed"] == 1).sum()))
print("저장 행 수:", len(result))
print("저장: outputs/valid_predictions.csv / test는 사용하지 않았습니다.")
