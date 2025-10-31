using Microsoft.ML.OnnxRuntime;
using ManySpeech.AudioSep.Model;
using System.Collections.Generic;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// Defines the contract for speech separation project implementations, 
    /// encapsulating model inference, metadata management, and audio processing parameters.
    /// </summary>
    internal interface ISepProj : IDisposable
    {
        /// <summary>
        /// Gets or sets the ONNX inference session for the main separation model.
        /// </summary>
        InferenceSession ModelSession { get; set; }

        /// <summary>
        /// Gets or sets custom metadata associated with the separation model.
        /// </summary>
        CustomMetadata CustomMetadata { get; set; }

        /// <summary>
        /// Gets or sets the length of audio chunks processed in each inference step (in samples).
        /// </summary>
        int ChunkLength { get; set; }

        /// <summary>
        /// Gets or sets the shift length between consecutive audio chunks (in samples).
        /// </summary>
        int ShiftLength { get; set; }

        /// <summary>
        /// Gets or sets the dimension of input features (e.g., mel-spectrogram bands).
        /// </summary>
        int FeatureDim { get; set; }

        /// <summary>
        /// Gets or sets the sample rate of the audio (in Hz).
        /// </summary>
        int SampleRate { get; set; }

        /// <summary>
        /// Gets or sets the number of audio channels.
        /// </summary>
        int Channels { get; set; }

        /// <summary>
        /// Processes a batch of model inputs through the separation model.
        /// </summary>
        /// <param name="modelInputs">List of input entities containing audio data and parameters.</param>
        /// <param name="statesList">Optional list of state arrays for incremental processing (e.g., cached features).</param>
        /// <param name="offset">Offset position in the audio stream for incremental processing.</param>
        /// <returns>List of output entities containing separated audio stems and metadata.</returns>
        List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]>? statesList = null, int offset = 0);

        /// <summary>
        /// Processes model outputs through the generator to refine or post-process separation results.
        /// </summary>
        /// <param name="modelOutputEntity">Input output entity from the main separation model.</param>
        /// <param name="batchSize">Number of items in the batch being processed.</param>
        /// <returns>List of refined output entities containing processed audio stems.</returns>
        List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1);
    }
}