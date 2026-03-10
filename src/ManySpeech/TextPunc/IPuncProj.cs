using Microsoft.ML.OnnxRuntime;
using ManySpeech.TextPunc.Model;
using System.Collections.Generic;

namespace ManySpeech.TextPunc
{
    /// <summary>
    /// Defines the contract for speech separation project implementations, 
    /// encapsulating model inference, metadata management, and audio processing parameters.
    /// </summary>
    internal interface IPuncProj : IDisposable
    {
        /// <summary>
        /// Gets or sets the ONNX inference session for the main separation model.
        /// </summary>
        InferenceSession ModelSession { get; set; }

        ///// <summary>
        ///// Gets or sets custom metadata associated with the separation model.
        ///// </summary>
        //CustomMetadata CustomMetadata { get; set; }

        ///// <summary>
        ///// Gets or sets the length of audio chunks processed in each inference step (in samples).
        ///// </summary>
        //int ChunkLength { get; set; }

        ///// <summary>
        ///// Gets or sets the shift length between consecutive audio chunks (in samples).
        ///// </summary>
        //int ShiftLength { get; set; }

        ///// <summary>
        ///// Gets or sets the dimension of input features (e.g., mel-spectrogram bands).
        ///// </summary>
        //int FeatureDim { get; set; }

        ///// <summary>
        ///// Gets or sets the sample rate of the audio (in Hz).
        ///// </summary>
        //int SampleRate { get; set; }

        ///// <summary>
        ///// Gets or sets the number of audio channels.
        ///// </summary>
        //int Channels { get; set; }

        
        /// <summary>
        /// Processes a non batch of model input through the separation model.
        /// </summary>
        /// <param name="modelInput">List of input entities containing tokens data and parameters.</param>
        PuncOutputEntity ModelProj(PuncInputEntity modelInput);
    }
}