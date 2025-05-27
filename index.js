//Variable global para el reconocedor de voz
let recognizer;
// Este método permite predecir palabras usando un modelo preentrenado
function predictWord() {
 // Array de palabras that que el reconocedor detecta.
 const words = recognizer.wordLabels();
 // Comienza a escuchar y procesa los puntajes de predicción
 recognizer.listen(({scores}) => {
   // Convierte el array de scores en objetos con palabra y score
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Ordena los scores de mayor a menor
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: 0.75});
}
// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
// Almacena los ejemplos de audio recolectados
let examples = [];
// Recolecta datos de audio para una etiqueta
function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 // Comienza a escuchar y guarda el espectrograma normalizado para entrenar
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES)); // Solo últimos 3 frames
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}
// Normaliza los valores del espectrograma
function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}
// Forma del input del modelo: 3 frames, 232 valores por frame, 1 canal
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
//Definimos variable global del modelo
let model;
// Entrena el modelo con los datos recolectados
async function train() {
  //Desactiva los demas botones mientras se entrena
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  // Entrena el modelo
 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       // Muestra la precisión y el número de epoch
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}
// Crea el modelo CNN con capas convolucionales y densas
function buildModel() {
 model = tf.sequential();
// Capa convolucional para detectar patrones temporales
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
   // Capa convolucional para detectar patrones temporales
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  // Aplana las salidas para alimentar a la capa densa
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
// Compila el modelo con Adam y entropía cruzada
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

// Activa o desactiva los botones de la interfaz
function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}
// Mueve el control deslizante en función del resultado del modelo
async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 document.getElementById('console').textContent = label;
 if (label == 2) {
   return;
 }
 // Cambia el valor del slider
 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}
//Boton listen, Escucha en tiempo real
function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Listen';
   return;
 }
 //Configuracion para el boton listen
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES)); // Normaliza
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]); // Prepara tensor
    const probs = model.predict(input); // Predice
    const predLabel = probs.argMax(1); // Saca la clase con mayor probabilidad
    await moveSlider(predLabel); // Mueve el slider
    tf.dispose([input, probs, predLabel]); // Libera memoria
  }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}
// Inicializa el reconocedor y el modelo al cargar la página
async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 // Carga modelo preentrenado
 await recognizer.ensureModelLoaded();
 // Construye el modelo personalizado
 buildModel();
}

app();