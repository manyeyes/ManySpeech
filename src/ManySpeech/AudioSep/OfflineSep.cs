// See https://github.com/manyeyes for more information
// Copyright (c) 2025 by manyeyes
using ManySpeech.AudioSep.Model;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// Offline audio separation processor that handles model loading, stream management,
    /// and audio separation operations.
    /// </summary>
    public class OfflineSep : IDisposable
    {
        private bool _disposed;
        private string[] _tokens = Array.Empty<string>();
        private string _mvnFilePath = string.Empty;
        private readonly ISepProj _sepProj;

        /// <summary>
        /// Initializes a new instance of the <see cref="OfflineSep"/> class.
        /// </summary>
        /// <param name="modelFilePath">Path to the main separation model file.</param>
        /// <param name="generatorFilePath">Path to the generator model file (optional).</param>
        /// <param name="configFilePath">Path to the configuration file (optional).</param>
        /// <param name="threadsNum">Number of threads to use for inference.</param>
        /// <exception cref="ArgumentException">Thrown when the model type is unsupported.</exception>
        public OfflineSep(string modelFilePath, string generatorFilePath = "", string configFilePath = "", int threadsNum = 1)
        {
            var sepModel = new SepModel(modelFilePath, generatorFilePath, configFilePath, threadsNum);
            var modelType = sepModel.ConfEntity?.model?.model_type?.ToLowerInvariant();

            _sepProj = modelType switch
            {
                "gtcrnse" => new SepProjOfGtcrnSe(sepModel),
                "spleeter" => new SepProjOfSpleeter(sepModel),
                "uvr" => new SepProjOfUvr(sepModel),
                _ => throw new ArgumentException($"Unsupported model type: {modelType}")
            };
        }

        /// <summary>
        /// Creates a new offline stream for processing audio.
        /// </summary>
        /// <returns>A new <see cref="OfflineStream"/> instance.</returns>
        public OfflineStream CreateOfflineStream()
        {
            return new OfflineStream(_mvnFilePath, _sepProj);
        }

        /// <summary>
        /// Gets separation results for a single offline stream.
        /// </summary>
        /// <param name="stream">The offline stream to process.</param>
        /// <returns>Separation results contained in an <see cref="OfflineSepResultEntity"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is null.</exception>
        public OfflineSepResultEntity GetResult(OfflineStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var streams = new List<OfflineStream> { stream };
            return GetResults(streams)[0];
        }

        /// <summary>
        /// Gets separation results for multiple offline streams.
        /// </summary>
        /// <param name="streams">List of offline streams to process.</param>
        /// <returns>List of separation results, one for each stream.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="streams"/> is null.</exception>
        public List<OfflineSepResultEntity> GetResults(List<OfflineStream> streams)
        {
            if (streams == null)
                throw new ArgumentNullException(nameof(streams));

            Forward(streams);
            return DecodeMulti(streams);
        }

        /// <summary>
        /// Processes audio data through the separation model.
        /// </summary>
        /// <param name="streams">List of streams containing audio data to process.</param>
        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
                return;

            var workingStreams = new List<OfflineStream>();
            var modelInputs = new List<ModelInputEntity>();
            var streamsToRemove = new List<OfflineStream>();

            // Prepare model inputs from valid streams
            foreach (var stream in streams)
            {
                var input = new ModelInputEntity
                {
                    SampleRate = stream.SampleRate,
                    Channels = stream.Channels,
                    Speech = stream.GetDecodeChunk()
                };

                if (input.Speech == null)
                {
                    streamsToRemove.Add(stream);
                    continue;
                }

                input.SpeechLength = input.Speech.Length;
                modelInputs.Add(input);
                workingStreams.Add(stream);
            }

            // Remove streams with no valid input
            foreach (var stream in streamsToRemove)
            {
                streams.Remove(stream);
            }

            if (modelInputs.Count == 0)
                return;

            try
            {
                // Run model inference
                var modelOutputs = _sepProj.ModelProj(modelInputs);
                var generatorOutputs = _sepProj.GeneratorProj(modelOutputs[0]);
                var finalOutputs = generatorOutputs ?? modelOutputs;

                // Associate results with their respective streams
                for (int i = 0; i < workingStreams.Count; i++)
                {
                    var stream = workingStreams[i];
                    stream.ModelOutputEntities ??= new List<ModelOutputEntity>();
                    stream.ModelOutputEntities.AddRange(finalOutputs);
                    stream.RemoveDecodedChunk();
                }
            }
            catch (Exception ex)
            {
                // Log exception (implement proper logging in production)
                Console.WriteLine($"Error during model inference: {ex.Message}");
            }
        }

        /// <summary>
        /// Flattens a list of float arrays into a single contiguous float array.
        /// </summary>
        /// <param name="arrays">List of float arrays to flatten.</param>
        /// <returns>Flattened float array.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="arrays"/> is null.</exception>
        public static float[] FlattenList(List<float[]> arrays)
        {
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            int totalLength = arrays.Sum(arr => arr.Length);
            var result = new float[totalLength];
            int offset = 0;

            foreach (var array in arrays)
            {
                Array.Copy(array, 0, result, offset, array.Length);
                offset += array.Length;
            }

            return result;
        }

        /// <summary>
        /// Decodes model outputs into separation results for multiple streams.
        /// </summary>
        /// <param name="streams">List of streams with model outputs to decode.</param>
        /// <returns>List of separation results.</returns>
        private List<OfflineSepResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            var results = new List<OfflineSepResultEntity>();

            try
            {
                foreach (var stream in streams)
                {
                    var result = new OfflineSepResultEntity
                    {
                        AudioId = stream.AudioId,
                        Channels = stream.Channels,
                        SampleRate = stream.SampleRate,
                        Stems = new Dictionary<string, float[]>()
                    };

                    if (stream.ModelOutputEntities == null || !stream.ModelOutputEntities.Any())
                    {
                        results.Add(result);
                        continue;
                    }

                    // Group outputs by stem name and flatten
                    var stemGroups = stream.ModelOutputEntities
                        .GroupBy(output => output.StemName)
                        .Where(group => !string.IsNullOrEmpty(group.Key));

                    foreach (var group in stemGroups)
                    {
                        var stemData = group.Select(output => output.StemContents)
                                           .Where(data => data != null)
                                           .ToList();

                        if (stemData.Count > 0)
                        {
                            result.Stems[group.Key] = FlattenList(stemData);
                        }
                    }

                    results.Add(result);
                }
            }
            catch (Exception ex)
            {
                // Log exception (implement proper logging in production)
                Console.WriteLine($"Error during decoding: {ex.Message}");
            }

            return results;
        }

        /// <summary>
        /// Disposes of an offline stream and releases its resources.
        /// </summary>
        /// <param name="stream">The stream to dispose.</param>
        public void DisposeOfflineStream(OfflineStream stream)
        {
            stream?.Dispose();
        }

        /// <summary>
        /// Releases the unmanaged resources used by the <see cref="OfflineSep"/> and optionally releases managed resources.
        /// </summary>
        /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    _sepProj?.Dispose();
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Releases all resources used by the <see cref="OfflineSep"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer for <see cref="OfflineSep"/>.
        /// </summary>
        ~OfflineSep()
        {
            Dispose(disposing: false);
        }
    }
}