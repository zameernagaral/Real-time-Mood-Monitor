import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';

const LABELS = ['Happy', 'Sad', 'Stressed'];
const videomood= document.getElementById('video');
const overlaymood = document.getElementById('overlay');
const ctx = overlaymood.getContext('2d');
const moodElement = document.getElementById('mood');




let mobilenet;
let classifiermood;
const SMOOTH_WINDOW = 10;
let predictionsBuffer = [];

async function initializeCamera() {
  const media = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false
  });
  videomood.srcObject = media;
  return new Promise(resolve => (videomood.onloadedmetadata = resolve));
}

async function loadFeatureExtractor() {
  mobilenet = await mobilenetModule.load({ version: 2, alpha: 1.0 });
}

async function loadClassifier() {
  try {
    classifiermood = await tf.loadLayersModel('/models/mood-stress-model.json');
    return true;
  } catch (error) {
    console.warn('Model load failed:', error);
    return false;
  }
}

function getFeatureVector() {
  const act = mobilenet.infer(videomood, 'global_average');
  const embedding = act.squeeze().dataSync();
  act.dispose();
  return embedding;
}

function renderOverlay(message) {
  ctx.clearRect(0, 0, overlaymood.width, overlaymood.height);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
  ctx.fillRect(0, overlaymood.height - 40, 240, 36);
  ctx.fillStyle = 'white';
  ctx.font = '16px Inter, sans-serif';
  ctx.fillText(message, 8, overlaymood.height - 14);
}

async function predictMood() {
  const features = getFeatureVector();
  const inputTensor = tf.tensor2d([features]);
  const output = classifiermood.predict(inputTensor);
  const scores = output.dataSync();

  predictionsBuffer.push(scores);
  if (predictionsBuffer.length > SMOOTH_WINDOW) predictionsBuffer.shift();

  const averaged = Array.from({ length: LABELS.length }, (_, i) =>
    predictionsBuffer.reduce((sum, p) => sum + p[i], 0) / predictionsBuffer.length
  );

  const maxIndex = averaged.indexOf(Math.max(...averaged));
  const mood = LABELS[maxIndex];
  const confidence = averaged[maxIndex];


  moodElement.textContent = `Mood: ${mood} (${(confidence * 100).toFixed(0)}%)`;
  moodElement.style.color = mood === 'Stressed' ? 'red' : mood === 'Happy' ? 'green' : 'white';
  renderOverlay(`Detected Mood: ${mood}`);

  inputTensor.dispose();
  output.dispose();
}

(async function main() {
  try {
    await initializeCamera();
    overlaymood.width = videomood.videoWidth || 640;
    overlaymood.height = videomood.videoHeight || 480;

    renderOverlay('Loading models, please wait...');
    await loadFeatureExtractor();
    const modelReady = await loadClassifier();

    if (!modelReady) {
      renderOverlay('Error: Unable to locate model file.');
      return;
    }

    document.querySelector('.video-container').classList.add('model-loaded');
    await videomood.play();

    renderOverlay('System Ready – analyzing expressions...');
    console.log('Mood detection initialized.');

    setInterval(predictMood, 300);
  } catch (err) {
    console.error('Startup failed:', err);
    renderOverlay(`Error: ${err.message}`);
  }
})();
