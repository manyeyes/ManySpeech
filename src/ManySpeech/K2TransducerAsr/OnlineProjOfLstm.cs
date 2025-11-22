// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using ManySpeech.K2TransducerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace ManySpeech.K2TransducerAsr
{
    internal class OnlineProjOfLstm : IOnlineProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OnlineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public OnlineProjOfLstm(OnlineModel onlineModel)
        {
            _encoderSession = onlineModel.EncoderSession;
            _decoderSession = onlineModel.DecoderSession;
            _joinerSession = onlineModel.JoinerSession;
            _blank_id = onlineModel.Blank_id;
            _sos_eos_id = onlineModel.Sos_eos_id;
            _unk_id = onlineModel.Unk_id;
            _featureDim = onlineModel.FeatureDim;

            _customMetadata = new OnlineCustomMetadata();
            _customMetadata = onlineModel.CustomMetadata;
            _chunkLength = _customMetadata.TT;
            _shiftLength = _customMetadata.Decode_chunk_len;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OnlineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public List<List<float[]>> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> state0 = new List<float[]>();
            List<float[]> state1 = new List<float[]>();
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            //计算尺寸
            for (int i = 0; i < num_encoders; i++)
            {
                int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                int d_model = _customMetadata.D_model;
                int rnn_hidden_size = _customMetadata.Rnn_hidden_size;
                //state0
                int state0_size = num_encoder_layers * batchSize * d_model;
                float[] state0_item = new float[state0_size];
                state0.Add(state0_item);
                //state1
                int state1_size = num_encoder_layers * batchSize * rnn_hidden_size;
                float[] state1_item = new float[state1_size];
                state1.Add(state1_item);
            }
            List<List<float[]>> statesList = new List<List<float[]>> { state0, state1 };
            return statesList;
        }

        public List<List<float[]>> stack_states_unittest(List<List<List<float[]>>> stateList)
        {
            List<List<float[]>> states = new List<List<float[]>>();
            states = stateList[0];
            return states;
        }

        public List<List<List<float[]>>> unstack_states_unittest(List<float[]> stateList)
        {
            List<List<List<float[]>>> xxx = new List<List<List<float[]>>>();
            List<List<float[]>> yyy = new List<List<float[]>>();
            List<float[]> zzz = new List<float[]>();
            zzz = new List<float[]>();
            zzz.Add(stateList[0]);
            zzz.Add(stateList[1]);
            yyy.Add(zzz);
            xxx.Add(yyy);
            return xxx;
        }
        public List<List<float[]>> stack_states(List<List<List<float[]>>> stateList)
        {
            //int num_encoders = _customMetadata.Num_encoder_layers.Length;
            int d_model = _customMetadata.D_model;
            int rnn_hidden_size = _customMetadata.Rnn_hidden_size;
            int batch_size = stateList.Count;
            Debug.Assert(stateList[0].Count % 2 == 0, "when stack_states, state_list[0] is 2x");
            int num_encoders = _customMetadata.Num_encoder_layers.Length;

            List<float[]> cache_state0 = new List<float[]>();
            List<float[]> cache_state1 = new List<float[]>();
            //state0
            List<List<float[]>> state0_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                state0_list.Add(stateList[n][0]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] state0 = new float[state0_list[0][i].Length * batch_size];
                int state0_item_length = state0_list[0][i].Length;
                int state0_axisnum = d_model;
                for (int x = 0; x < state0_item_length / state0_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] state0_avg_item = state0_list[n][i];
                        Array.Copy(state0_avg_item, x * state0_axisnum, state0, (x * batch_size + n) * state0_axisnum, state0_axisnum);
                    }
                }
                cache_state0.Add(state0);
            }
            //state1
            List<List<float[]>> state1_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                state1_list.Add(stateList[n][1]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] state1 = new float[state1_list[0][i].Length * batch_size];
                int state1_item_length = state1_list[0][i].Length;
                int state1_axisnum = rnn_hidden_size;
                for (int x = 0; x < state1_item_length / state1_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] state1_avg_item = state1_list[n][i];
                        Array.Copy(state1_avg_item, x * state1_axisnum, state1, (x * batch_size + n) * state1_axisnum, state1_axisnum);
                    }
                }
                cache_state1.Add(state1);
            }
            List<List<float[]>> states = new List<List<float[]>> { cache_state0, cache_state1 };
            return states;
        }

        public List<List<List<float[]>>> unstack_states(List<float[]> encoder_out_states)
        {
            int d_model = _customMetadata.D_model;
            int rnn_hidden_size = _customMetadata.Rnn_hidden_size;
            List<List<List<float[]>>> statesList = new List<List<List<float[]>>>();
            Debug.Assert(encoder_out_states.Count % 2 == 0, "when stack_states, encoder_out_states[0] is 2x");
            int batch_size = encoder_out_states[0].Length / _customMetadata.Num_encoder_layers[0] / d_model;
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            for (int i = 0; i < batch_size; i++)
            {
                List<float[]> state0 = new List<float[]>();
                List<float[]> state1 = new List<float[]>();
                for (int j = 0; j < 2; j++)
                {
                    for (int m = 0; m < num_encoders; m++)
                    {
                        int num_encoder_layers = _customMetadata.Num_encoder_layers[m];
                        float[] item = encoder_out_states[j * num_encoders + m];
                        int n = 1;
                        switch (j)
                        {
                            case 0:
                                //state0
                                int state0_axisnum = d_model;
                                int state0_size = num_encoder_layers * n * d_model;
                                float[] state0_item = new float[state0_size];
                                for (int k = 0; k < state0_size / state0_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / state0_size * k + i) * state0_axisnum, state0_item, k * state0_axisnum, state0_axisnum);
                                }
                                //Array.Copy(item, state0_size*i, state0_item, state0_size * i, state0_size);
                                state0.Add(state0_item);
                                break;
                            case 1:
                                //state1
                                int state1_axisnum = rnn_hidden_size;
                                int state1_size = num_encoder_layers * n * rnn_hidden_size;
                                float[] state1_item = new float[state1_size];
                                for (int k = 0; k < state1_size / state1_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / state1_size * k + i) * state1_axisnum, state1_item, k * state1_axisnum, state1_axisnum);
                                }
                                //Array.Copy(item, state1_size * i, state1_item, state1_size * i, state1_size);
                                state1.Add(state1_item);
                                break;
                        }
                    }
                }
                List<List<float[]>> states = new List<List<float[]>> { state0, state1 };

                statesList.Add(states);

            }
            return statesList;
        }

        public EncoderOutputEntity EncoderProj(List<OnlineInputEntity> modelInputs, int batchSize, List<List<float[]>> statesList)
        {
            OnlineCustomMetadata onlineCustomMetadata = _customMetadata;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            try
            {
                var container = new List<NamedOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "x")
                    {
                        int[] dim = new int[] { batchSize, padSequence.Length / FeatureDim / batchSize, FeatureDim };
                        var tensor = new DenseTensor<float>(padSequence, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                }
                for (int i = 0; i < statesList.Count; i++)
                {
                    int d_model = _customMetadata.D_model;
                    int rnn_hidden_size = _customMetadata.Rnn_hidden_size;
                    List<float[]> items = statesList[i];
                    for (int m = 0; m < items.Count; m++)
                    {
                        int num_encoder_layers = _customMetadata.Num_encoder_layers[m];
                        var state = items[m];
                        string name = "";
                        int[] dim = new int[1];
                        switch (i)
                        {
                            case 0:
                                name = "state";
                                dim = new int[] { num_encoder_layers, batchSize, d_model };
                                name = name + i.ToString();
                                var tensor_len = new DenseTensor<float>(state, dim, false);
                                container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_len));
                                break;
                            case 1:
                                name = "state";
                                dim = new int[] { num_encoder_layers, batchSize, rnn_hidden_size };
                                name = name + i.ToString();
                                var tensor_avg = new DenseTensor<float>(state, dim, false);
                                container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_avg));
                                break;
                        }
                    }
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.encoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
                    List<float[]> states = new List<float[]>();
                    for (int i = 1; i < encoderResultsArray.Length; i++)
                    {
                        float[] item;
                        if (encoderResultsArray[i].ElementType.ToString() == typeof(System.Int64).Name)
                        {
                            item = encoderResultsArray[i].AsEnumerable<Int64>().Select(x => (float)Convert.ToDouble(x)).ToArray();
                        }
                        else
                        {
                            item = encoderResultsArray[i].AsEnumerable<float>().ToArray();
                        }
                        states.Add(item);
                    }
                    encoderOutput.encoder_out_states = states;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }
        public DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize)
        {
            int contextSize = _customMetadata.Context_size;
            DecoderOutputEntity decoderOutput = new DecoderOutputEntity();
            if (decoder_input == null)
            {
                Int64[] hyp = new Int64[] { -1, _blank_id };
                decoder_input = hyp;
                if (batchSize > 1)
                {
                    decoder_input = new Int64[contextSize * batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        Array.Copy(hyp, 0, decoder_input, i * contextSize, contextSize);
                    }
                }

            }
            var decoder_container = new List<NamedOnnxValue>();
            int[] dim = new int[] { decoder_input.Length / contextSize, contextSize };
            var decoder_input_tensor = new DenseTensor<Int64>(decoder_input, dim, false);
            decoder_container.Add(NamedOnnxValue.CreateFromTensor<Int64>("y", decoder_input_tensor));
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
            decoderResults = _decoderSession.Run(decoder_container);
            if (decoderResults != null)
            {
                var encoderResultsArray = decoderResults.ToArray();
                decoderOutput.decoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
            }
            return decoderOutput;
        }

        public JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {
            int joinerDim = _customMetadata.Joiner_dim;
            var inputMeta = _joinerSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "encoder_out")
                {
                    int[] dim = new int[] { encoder_out.Length / joinerDim, joinerDim };
                    var tensor = new DenseTensor<float>(encoder_out, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "decoder_out")
                {
                    int[] dim = new int[] { decoder_out.Length / joinerDim, joinerDim };
                    var tensor = new DenseTensor<float>(decoder_out, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> joinerResults = null;
            joinerResults = _joinerSession.Run(container);
            JoinerOutputEntity joinerOutput = new JoinerOutputEntity();
            var joinerResultsArray = joinerResults.ToArray();
            joinerOutput.Logit = joinerResultsArray[0].AsEnumerable<float>().ToArray();
            joinerOutput.Logits = joinerResultsArray[0].AsTensor<float>();
            return joinerOutput;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_encoderSession != null)
                    {
                        _encoderSession.Dispose();
                    }
                    if (_decoderSession != null)
                    {
                        _decoderSession.Dispose();
                    }
                    if (_joinerSession != null)
                    {
                        _joinerSession.Dispose();
                    }
                    if (_customMetadata != null)
                    {
                        _customMetadata = null;
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~OnlineProjOfLstm()
        {
            Dispose(_disposed);
        }
    }
}
