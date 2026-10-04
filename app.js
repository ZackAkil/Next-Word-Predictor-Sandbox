// Teachable Next-Word Predictor Logic

let ngramMap = {};
let vocabulary = [];
let currentContextSize = 1;

// Sample Presets
const SAMPLE_ANIMALS = [
  "the cat sat on the mat",
  "the cat ate the fish",
  "the dog sat on the log",
  "the dog chased the cat",
  "the bird sang on the tree"
].join("\n");

const SAMPLE_RHYME = [
  "twinkle twinkle little star",
  "how I wonder what you are",
  "up above the world so high",
  "like a diamond in the sky"
].join("\n");

// DOM Elements
const trainingInput = document.getElementById("training-input");
const contextSizeSelect = document.getElementById("context-size");
const trainBtn = document.getElementById("train-btn");
const modelTableContainer = document.getElementById("model-table-container");

const promptInput = document.getElementById("prompt-input");
const tokensCountInput = document.getElementById("tokens-count");
const samplingModeSelect = document.getElementById("sampling-mode");
const predictBtn = document.getElementById("predict-btn");

const presetAnimalsBtn = document.getElementById("preset-animals");
const presetRhymeBtn = document.getElementById("preset-rhyme");
const clearDataBtn = document.getElementById("clear-data");

const generationResult = document.getElementById("generation-result");
const generatedTextDisplay = document.getElementById("generated-text-display");
const traceContainer = document.getElementById("trace-container");
const strategyExplainer = document.getElementById("strategy-explainer");

const SAMPLING_EXPLANATIONS = {
  "probability-sampled": {
    title: "Probability-Weighted Sampling",
    desc: "The model samples tokens randomly, weighted according to their calculated probability distribution (e.g., a token with 70% probability has a 70% chance of selection). Produces natural, human-like language generation."
  },
  "greedy": {
    title: "Greedy Sampling (Top Choice)",
    desc: "The model deterministically selects the single token with the highest probability at each step. Fast and predictable, but prone to getting stuck in repetitive loops."
  },
  "uniform-random": {
    title: "Pure Random Sampling",
    desc: "The model ignores probabilities completely and selects uniformly at random from all valid candidate tokens. Produces chaotic, highly unpredictable output."
  }
};

function updateStrategyExplainer() {
  const selectedMode = samplingModeSelect.value;
  const info = SAMPLING_EXPLANATIONS[selectedMode] || SAMPLING_EXPLANATIONS["probability-sampled"];
  strategyExplainer.innerHTML = `
    <h4>${info.title}</h4>
    <p>${info.desc}</p>
  `;
}

// Initialize with default preset
trainingInput.value = SAMPLE_ANIMALS;

// Tokenize text into words
function tokenize(text) {
  if (!text) return [];
  // Basic clean tokenizer: lowercase, strip extra whitespace
  return text
    .toLowerCase()
    .replace(/[^\w\s']/g, "")
    .trim()
    .split(/\s+/)
    .filter(token => token.length > 0);
}

// Train the N-gram Model
function trainModel() {
  const text = trainingInput.value;
  currentContextSize = parseInt(contextSizeSelect.value, 10);

  const lines = text.split("\n");
  ngramMap = {};
  const vocabSet = new Set();

  lines.forEach(line => {
    const tokens = tokenize(line);
    tokens.forEach(t => vocabSet.add(t));

    for (let i = 0; i <= tokens.length - currentContextSize - 1; i++) {
      const contextTokens = tokens.slice(i, i + currentContextSize);
      const contextKey = contextTokens.join(" ");
      const nextToken = tokens[i + currentContextSize];

      if (!ngramMap[contextKey]) {
        ngramMap[contextKey] = {};
      }
      ngramMap[contextKey][nextToken] = (ngramMap[contextKey][nextToken] || 0) + 1;
    }
  });

  vocabulary = Array.from(vocabSet);
  renderModelTable();
}

// Render Model Probability Table
function renderModelTable() {
  const contextKeys = Object.keys(ngramMap);

  if (contextKeys.length === 0) {
    modelTableContainer.innerHTML = `<p class="placeholder-text">Model not trained yet. Provide training data and click "Train Model".</p>`;
    return;
  }

  let html = `
    <table>
      <thead>
        <tr>
          <th>Context (${currentContextSize} token${currentContextSize > 1 ? 's' : ''})</th>
          <th>Next Word Probabilities</th>
        </tr>
      </thead>
      <tbody>
  `;

  contextKeys.forEach(contextKey => {
    const nextWordsObj = ngramMap[contextKey];
    const totalCount = Object.values(nextWordsObj).reduce((a, b) => a + b, 0);

    const candidates = Object.entries(nextWordsObj)
      .map(([word, count]) => ({
        word,
        count,
        prob: (count / totalCount)
      }))
      .sort((a, b) => b.prob - a.prob);

    const pillsHtml = candidates.map(c => `
      <span class="prob-pill">
        <strong>${c.word}</strong>
        <span class="prob-val">${(c.prob * 100).toFixed(0)}%</span>
      </span>
    `).join("");

    html += `
      <tr>
        <td><span class="context-tag">${contextKey}</span></td>
        <td>${pillsHtml}</td>
      </tr>
    `;
  });

  html += `</tbody></table>`;
  modelTableContainer.innerHTML = html;
}

// Perform Inference / Generation
function predictNextWords() {
  if (Object.keys(ngramMap).length === 0) {
    alert("Please train the model first by clicking 'Train Model'!");
    return;
  }

  const rawPrompt = promptInput.value;
  const numTokensToPredict = parseInt(tokensCountInput.value, 10) || 3;
  const samplingMode = samplingModeSelect.value;

  let currentTokens = tokenize(rawPrompt);
  const promptTokenCount = currentTokens.length;

  if (currentTokens.length < currentContextSize) {
    alert(`Prompt needs at least ${currentContextSize} token(s) to match the current context size.`);
    return;
  }

  const generatedTokens = [];
  const traces = [];

  for (let step = 1; step <= numTokensToPredict; step++) {
    // Get context from last N tokens
    const contextTokens = currentTokens.slice(-currentContextSize);
    const contextKey = contextTokens.join(" ");

    let candidatesObj = ngramMap[contextKey];
    let isFallback = false;

    // Fallback strategy if exact context is not found in training data
    if (!candidatesObj) {
      // Try shorter context if N=2
      if (currentContextSize === 2 && contextTokens.length > 1) {
        const shorterKey = contextTokens[contextTokens.length - 1];
        candidatesObj = ngramMap[shorterKey];
      }
      if (!candidatesObj) {
        isFallback = true;
      }
    }

    let chosenWord = "";
    let candidatesList = [];

    if (!isFallback && candidatesObj) {
      const totalCount = Object.values(candidatesObj).reduce((a, b) => a + b, 0);
      candidatesList = Object.entries(candidatesObj)
        .map(([word, count]) => ({
          word,
          count,
          prob: count / totalCount
        }))
        .sort((a, b) => b.prob - a.prob);

      if (samplingMode === "greedy") {
        // Top probability token
        chosenWord = candidatesList[0].word;
      } else if (samplingMode === "uniform-random") {
        // Pure uniform random among candidate tokens
        const randIndex = Math.floor(Math.random() * candidatesList.length);
        chosenWord = candidatesList[randIndex].word;
      } else {
        // Probability-weighted random sample
        const rand = Math.random();
        let cumulative = 0;
        for (const cand of candidatesList) {
          cumulative += cand.prob;
          if (rand <= cumulative) {
            chosenWord = cand.word;
            break;
          }
        }
        if (!chosenWord) chosenWord = candidatesList[0].word;
      }
    } else {
      // Unigram / random vocabulary fallback
      if (vocabulary.length > 0) {
        chosenWord = vocabulary[Math.floor(Math.random() * vocabulary.length)];
        candidatesList = [{ word: chosenWord, count: 1, prob: 1.0 }];
      } else {
        chosenWord = "[UNKNOWN]";
      }
    }

    currentTokens.push(chosenWord);
    generatedTokens.push(chosenWord);

    traces.push({
      step,
      contextKey,
      candidates: candidatesList,
      chosenWord,
      isFallback
    });
  }

  // Display results
  renderGenerationResults(currentTokens, promptTokenCount, traces);
}

// Render Results and Traces
function renderGenerationResults(allTokens, promptTokenCount, traces) {
  generationResult.classList.remove("hidden");

  const promptPart = allTokens.slice(0, promptTokenCount).join(" ");
  const generatedPart = allTokens.slice(promptTokenCount).map(t => `<span class="new-token">${t}</span>`).join(" ");

  generatedTextDisplay.innerHTML = `<span class="prompt-part">${promptPart} </span>${generatedPart}`;

  let traceHtml = "";
  traces.forEach(t => {
    const candItems = t.candidates.map(c => `
      <span class="cand-item ${c.word === t.chosenWord ? 'chosen' : ''}">
        ${c.word}: ${(c.prob * 100).toFixed(0)}%
      </span>
    `).join("");

    traceHtml += `
      <div class="trace-step">
        <div class="trace-step-num">Step ${t.step}: Context = "${t.contextKey}" ${t.isFallback ? '(Fallback context, used random word)' : ''}</div>
        <div class="trace-details">
          <div>Token Probabilities:</div>
          <div class="candidates-list">${candItems}</div>
          <div style="margin-top: 0.4rem;">Selected Token: <strong>${t.chosenWord}</strong></div>
        </div>
      </div>
    `;
  });

  traceContainer.innerHTML = traceHtml;
}

// Event Listeners
trainBtn.addEventListener("click", trainModel);
predictBtn.addEventListener("click", predictNextWords);

presetAnimalsBtn.addEventListener("click", () => {
  trainingInput.value = SAMPLE_ANIMALS;
  trainModel();
});

presetRhymeBtn.addEventListener("click", () => {
  trainingInput.value = SAMPLE_RHYME;
  trainModel();
});

clearDataBtn.addEventListener("click", () => {
  trainingInput.value = "";
  ngramMap = {};
  renderModelTable();
});

samplingModeSelect.addEventListener("change", updateStrategyExplainer);

// Initial setup on page load
// trainModel();
updateStrategyExplainer();
