# Reconocimiento de Audio con TensorFlow.js 

Este proyecto es una implementación basada en el [Codelab oficial de TensorFlow.js: Reconocimiento de audio mediante aprendizaje por transferencia](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab?hl=es_419#0), adaptado y ajustado para entrenar un modelo personalizado que puede reconocer sonidos específicos y mover un control deslizante (`slider`) en el navegador dependiendo de los comandos que reconozca.

##  Descripción

En este proyecto:

- Carga de un modelo previamente entrenado que reconoce comandos de voz.
  Utilizacion del  micrófono para recolectar sonidos personalizados.
- Entrenamiento de una red neuronal simple para reconocer esos sonidos en tiempo real.

##  Cómo ejecutar este proyecto

1. **Cloná el repositorio**:

```bash
git clone https://github.com/Fleitaselene-dev/codelab_reco_audio_aprendizaje.git
cd codelab_reco_audio_aprendizaje
```
2. Abrí el archivo index.html en tu navegador.
3. Recolectá datos:
   - Usá los botones `Left`, `Right` y `Noise` para grabar sonidos.
   - Mantené presionado cada botón durante 3 a 4 segundos para grabar correctamente.
   - Hacelo varias veces hasta reunir al menos 150 ejemplos por clase.
4. Entrená el modelo:
   - Presioná el botón `Entrenar`.
   - El entrenamiento comenzará y deberías ver la precisión del modelo (accuracy).
   - Se recomienda alcanzar más del 90% de precisión. Si no lo lográs, recopilá más datos.
5. Probá el modelo:
   - Presioná el botón `Listen`.
   - Emití sonidos similares a los grabados y observá cómo se mueve el control deslizante (slider) según los comandos detectados.
