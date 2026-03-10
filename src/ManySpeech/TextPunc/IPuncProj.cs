using ManySpeech.TextPunc.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.TextPunc
{
    /// <summary>
    /// Defines the contract for text punc project implementations, 
    /// encapsulating model inference, metadata management, and text processing parameters.
    /// </summary>
    internal interface IPuncProj : IDisposable
    {
        /// <summary>
        /// Gets or sets the ONNX inference session for the main punc model.
        /// </summary>
        InferenceSession ModelSession { get; set; }
        
        /// <summary>
        /// Processes a non batch of model input through the punc model.
        /// </summary>
        /// <param name="modelInput">List of input entities containing tokens data and parameters.</param>
        PuncOutputEntity ModelProj(PuncInputEntity modelInput);
    }
}