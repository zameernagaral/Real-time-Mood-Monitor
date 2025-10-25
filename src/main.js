import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';

// Simple label map: mood labels and stress label derived from mood
const LABELS = ['happy','sad','stressed'];

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const btnHappy = document.getElementById('collect-happy');
const btnSad = document.getElementById('collect-sad');
const btnStressed = document.getElementById('collect-stressed');
const btnTrain = document.getElementById('train');
const btnSave = document.getElementById('save');
const moodEl = document.getElementById('mood');
const stressEl = document.getElementById('stress');

let mobilenet;
let classifierModel; // small dense model we train
let collectingLabel = null;
let samples = []; // {embedding: Float32Array, labelIndex}
const EMBED_SIZE = 1280; // mobilenet v2 embedding size — must match saved model input
const SMOOTHING_WINDOW = 10;
let recentPredictions = [];

// ...existing code...
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}, audio:false});
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

async function loadMobileNet() {
  // load Mobilenet V2 from tfjs-models; use version with stride; will be used for inference and feature extraction
  mobilenet = await mobilenetModule.load({version:2, alpha:1.0});
  console.log('MobileNet loaded');
}
async function loadSavedModel() {
  try {
    classifierModel = await tf.loadLayersModel('/models/mood-stress-model.json');
    console.log('Loaded saved classifier model');
    return true;
  } catch (e) {
    console.warn('No saved model found:', e.message);
    return false;
  }
}
async function validateSavedModel() {
  try {
    const m = await tf.loadLayersModel('/models/mood-stress-model.json');
    console.log('Saved model input shape:', m.inputs[0].shape);
    await m.save; // no-op to avoid lint warning
    m.dispose();
    return true;
  } catch (err) {
    console.error('Failed to load saved model:', err);
    return false;
  }
}

function captureEmbedding() {
  // mobilenet.infer accepts an image (HTMLVideoElement) and a pooling strategy.
  // 'conv_preds' returns the logits; use 'global_average' or default to get embeddings.
  // We'll use intermediate activation vector (global_pool) as features.
  const activation = mobilenet.infer(video, 'global_average');
  // activation is a tf.Tensor of shape [1, featureSize]
  const emb = activation.squeeze().dataSync(); // Float32Array
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

btnHappy.onclick = () => collectingLabel = 0;
btnSad.onclick = () => collectingLabel = 1;
btnStressed.onclick = () => collectingLabel = 2;

btnTrain.onclick = async () => {
  if (samples.length === 0) { alert('No samples collected'); return; }
  await buildAndTrainModel();
};

btnSave.onclick = async () => {
  if (!classifierModel) { alert('Train model first'); return; }
  await classifierModel.save('downloads://mood-stress-model');
};

// collect samples while button pressed: sample each 300ms while collectingLabel != null
video.addEventListener('play', () => {
  const interval = setInterval(async () => {
    if (collectingLabel !== null) {
      const emb = captureEmbedding();
      samples.push({embedding: Array.from(emb), label: collectingLabel});
      console.log('Collected', samples.length, 'samples');
      drawOverlay(`Collected ${samples.length} (${LABELS[collectingLabel]})`);
    } else if (classifierModel) {
      // Always run prediction when not collecting
      await runLivePrediction();
    }
  }, 300);
  video.addEventListener('pause', () => clearInterval(interval));
});

// stop collection when mouseup anywhere
window.addEventListener('mouseup', () => { collectingLabel = null; });

// build and train classifier on embeddings
async function buildAndTrainModel() {
    tf.util.shuffle(samples);
    const xs = tf.tensor2d(samples.map(s => s.embedding));
    const ys = tf.tensor1d(samples.map(s => s.label), 'int32');
    const ysOneHot = tf.oneHot(ys, LABELS.length);

    // Deeper architecture with regularization
    classifierModel = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [EMBED_SIZE],
                units: 512,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.01})
            }),
            tf.layers.dropout({rate: 0.4}),
            tf.layers.dense({
                units: 256,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.01})
            }),
            tf.layers.dropout({rate: 0.3}),
            tf.layers.dense({
                units: LABELS.length,
                activation: 'softmax'
            })
        ]
    });
     classifierModel.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

  await classifierModel.fit(xs, ysOneHot, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: [
            tf.callbacks.earlyStopping({
                monitor: 'val_loss',
                patience: 10
            })
        ]
    });


  xs.dispose();
  ysOneHot.dispose();

  alert('Training finished. Start moving/expressing in front of camera to see live prediction.');
}

// live run: capture embedding, run classifier, update UI
async function runLivePrediction() {
    const embArr = captureEmbedding();
    const x = tf.tensor2d([embArr]);
    const logits = classifierModel.predict(x);
    const probs = logits.dataSync();
    
    // Add to recent predictions
    recentPredictions.push(probs);
    if (recentPredictions.length > SMOOTHING_WINDOW) {
        recentPredictions.shift();
    }
    
    // Average recent predictions
    const smoothedProbs = Array.from({length: LABELS.length}, (_, i) => 
        recentPredictions.reduce((sum, p) => sum + p[i], 0) / recentPredictions.length
    );
    
    const idx = smoothedProbs.indexOf(Math.max(...smoothedProbs));
    const label = LABELS[idx];
    const conf = smoothedProbs[idx];

    moodEl.textContent = `Mood: ${label} (${(conf*100).toFixed(0)}%)`;
    stressEl.textContent = `Stress: ${(smoothedProbs[2]*100).toFixed(0)}%`;
    
    x.dispose();
    logits.dispose();
}
// init
(async function init(){
  try {
    await setupCamera();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    
    // Load both models before starting
    await Promise.all([
      loadMobileNet(),
      loadSavedModel(),
      validateSavedModel()
    ]);

    if (!classifierModel) {
      alert('No saved model found! Please train and save a model first.');
      return;
    }

    console.log('Ready for live predictions');
    // Start video to trigger predictions
    await video.play();
    
  } catch (err) {
    console.error('Initialization failed:', err);
    alert('Error starting app: ' + err.message);
  }
})();