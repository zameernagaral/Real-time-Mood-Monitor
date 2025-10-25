import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';

const LABELS = ['Happy','Sad','Stressed'];

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const moodEl = document.getElementById('mood');
const stressEl = document.getElementById('stress');
const themeToggle = document.getElementById('theme-toggle');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
  document.documentElement.setAttribute('data-theme', savedTheme);
} else if (prefersDarkScheme.matches) {
  document.documentElement.setAttribute('data-theme', 'light');
}

themeToggle.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
});

let mobilenet;
let classifierModel;
const EMBED_SIZE = 1280;
const SMOOTHING_WINDOW = 10;
let recentPredictions = [];

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}, audio:false});
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

async function loadMobileNet() {
  mobilenet = await mobilenetModule.load({version:2, alpha:1.0});
  console.log('MobileNet loaded');
}

async function loadSavedModel() {
  try {
    classifierModel = await tf.loadLayersModel('/models/mood-stress-model.json');
    console.log('Model loaded successfully');
    return true;
  } catch (e) {
    console.warn('Failed to load model:', e.message);
    return false;
  }
}

function captureEmbedding() {
  const activation = mobilenet.infer(video, 'global_average');
  const emb = activation.squeeze().dataSync();
  activation.dispose();
  return emb;
}

function drawOverlay(text) {
  ctx.clearRect(0,0,overlay.width,overlay.height);
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, overlay.height - 40, 200, 36);
  ctx.fillStyle = 'white';
  ctx.font = '16px sans-serif';
  ctx.fillText(text, 8, overlay.height - 14);
}


async function runLivePrediction() {
    const embArr = captureEmbedding();
    const x = tf.tensor2d([embArr]);
    const logits = classifierModel.predict(x);
    const probs = logits.dataSync();
    
    recentPredictions.push(probs);
    if (recentPredictions.length > SMOOTHING_WINDOW) {
        recentPredictions.shift();
    }
    
    const smoothedProbs = Array.from({length: LABELS.length}, (_, i) => 
        recentPredictions.reduce((sum, p) => sum + p[i], 0) / recentPredictions.length
    );
    
    const idx = smoothedProbs.indexOf(Math.max(...smoothedProbs));
    const label = LABELS[idx];
    const conf = smoothedProbs[idx];
    if(label === 'Stressed') {
        moodEl.style.color = 'red';
    }
    else if(label === "Happy") {
        moodEl.style.color = 'green';
    }
    moodEl.textContent = `Mood: ${label} (${(conf*100).toFixed(0)}%)`;
  
    moodEl.setAttribute('data-mood', label);
    drawOverlay(`Current Mood: ${label}`);
    
    x.dispose();
    logits.dispose();
}


(async function init(){
  try {
    await setupCamera();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    
    drawOverlay('Loading models...');
    
    await loadMobileNet();
    const modelLoaded = await loadSavedModel();

    if (!modelLoaded) {
      drawOverlay('Error: Model not found');
      return;
    }

    console.log('Ready for predictions');
    drawOverlay('Ready! Analyzing your expressions...');
  
document.querySelector('.video-container').classList.add('model-loaded');
    
    await video.play();
    
    
    setInterval(runLivePrediction, 300);
    
  } catch (err) {
    console.error('Initialization failed:', err);
    drawOverlay('Error: ' + err.message);
  }
})();