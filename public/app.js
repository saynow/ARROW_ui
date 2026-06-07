const form = document.querySelector("#prediction-form");
const subtypeInput = document.querySelector("#subtype");
const subtypeOptions = document.querySelector("#subtype-options");
const heavyInput = document.querySelector("#heavy-sequence");
const lightInput = document.querySelector("#light-sequence");
const heavyCount = document.querySelector("#heavy-count");
const lightCount = document.querySelector("#light-count");
const predictButton = document.querySelector("#predict-button");
const formError = document.querySelector("#form-error");
const modelStatus = document.querySelector("#model-status");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");

function sequenceLength(value) {
  return value.replace(/\s/g, "").length;
}

function updateSequenceCount(input, output) {
  output.textContent = `${sequenceLength(input.value)} aa`;
}

heavyInput.addEventListener("input", () => updateSequenceCount(heavyInput, heavyCount));
lightInput.addEventListener("input", () => updateSequenceCount(lightInput, lightCount));

async function loadMetadata() {
  try {
    const response = await fetch("/api/metadata");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "모델 정보를 불러오지 못했습니다.");
    }

    subtypeOptions.replaceChildren(
      ...data.subtypes.map((subtype) => {
        const option = document.createElement("option");
        option.value = subtype;
        return option;
      }),
    );
    modelStatus.classList.add("ready");
    modelStatus.lastElementChild.textContent =
      `모델 준비 완료 · 학습 데이터 ${data.training_count}개`;
  } catch (error) {
    modelStatus.lastElementChild.textContent = error.message;
  }
}

function scoreDescription(score) {
  if (score >= 8) return "모델이 높은 ADC 적합도 점수를 예측했습니다.";
  if (score >= 5) return "모델이 중간 수준의 ADC 적합도 점수를 예측했습니다.";
  return "모델이 낮은 ADC 적합도 점수를 예측했습니다.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  predictButton.disabled = true;
  predictButton.classList.add("loading");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subtype: subtypeInput.value,
        heavy_sequence: heavyInput.value,
        light_sequence: lightInput.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "예측에 실패했습니다.");
    }

    emptyResult.hidden = true;
    resultContent.hidden = false;
    document.querySelector("#score-value").textContent = data.adc_score.toFixed(3);
    document.querySelector("#score-caption").textContent =
      scoreDescription(data.adc_score);
    document.querySelector("#result-subtype").textContent = data.subtype;
    document.querySelector("#result-hcdr3").textContent = data.hcdr3;
    document.querySelector("#result-lcdr3").textContent = data.lcdr3;

    const scoreFill = document.querySelector("#score-fill");
    scoreFill.style.width = "0";
    requestAnimationFrame(() => {
      scoreFill.style.width = `${data.adc_score * 10}%`;
    });
  } catch (error) {
    formError.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.classList.remove("loading");
  }
});

loadMetadata();
