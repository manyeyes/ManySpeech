// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using ManySpeech.K2TransducerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace ManySpeech.K2TransducerAsr
{
    internal class OnlineProjOfZipformer : IOnlineProj, IDisposable
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
        private int _chunkLength = 0;//39;
        private int _shiftLength = 0;//32;
        public OnlineProjOfZipformer(OnlineModel onlineModel)
        {
            _encoderSession = onlineModel.EncoderSession;
            _decoderSession = onlineModel.DecoderSession;
            _joinerSession = onlineModel.JoinerSession;
            _blank_id = onlineModel.Blank_id;
            _sos_eos_id = onlineModel.Sos_eos_id;
            _unk_id = onlineModel.Unk_id;
            _featureDim = onlineModel.FeatureDim;
            _sampleRate = onlineModel.SampleRate;

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
            List<float[]> cached_len = new List<float[]>();
            List<float[]> cached_avg = new List<float[]>();
            List<float[]> cached_key = new List<float[]>();
            List<float[]> cached_val = new List<float[]>();
            List<float[]> cached_val2 = new List<float[]>();
            List<float[]> cached_conv1 = new List<float[]>();
            List<float[]> cached_conv2 = new List<float[]>();
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            //TODO 改为计算尺寸
            //int cached_len_size = 0;
            //int cached_avg_size = 0;
            //int cached_key_size = 0;
            //int cached_val_size = 0;
            //int cached_val2_size = 0;
            //int cached_conv1_size = 0;
            //int cached_conv2_size = 0;
            for (int i = 0; i < num_encoders; i++)
            {
                int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                //cached_len
                int cached_len_size = num_encoder_layers * batchSize;
                float[] cached_len_item = new float[cached_len_size];
                cached_len.Add(cached_len_item);
                //cached_avg
                int cached_avg_size = num_encoder_layers * batchSize * _customMetadata.Encoder_dims[i];
                float[] cached_avg_item = new float[cached_avg_size];
                cached_avg.Add(cached_avg_item);
                //cached_key
                int cached_key_size = num_encoder_layers * _customMetadata.Left_context_len[i] * batchSize * _customMetadata.Attention_dims[i];
                float[] cached_key_item = new float[cached_key_size];
                cached_key.Add(cached_key_item);
                //cached_val
                int cached_val_size = num_encoder_layers * _customMetadata.Left_context_len[i] * batchSize * _customMetadata.Attention_dims[i] / 2;
                float[] cached_val_item = new float[cached_val_size];
                cached_val.Add(cached_val_item);
                //cached_val2
                int cached_val2_size = num_encoder_layers * _customMetadata.Left_context_len[i] * batchSize * _customMetadata.Attention_dims[i] / 2;
                float[] cached_val2_item = new float[cached_val2_size];
                cached_val2.Add(cached_val2_item);
                //cached_conv1
                int cached_conv1_size = num_encoder_layers * batchSize * _customMetadata.Encoder_dims[i] * (_customMetadata.Cnn_module_kernels[i] - 1);
                float[] cached_conv1_item = new float[cached_conv1_size];
                cached_conv1.Add(cached_conv1_item);
                //cached_conv2
                int cached_conv2_size = num_encoder_layers * batchSize * _customMetadata.Encoder_dims[i] * (_customMetadata.Cnn_module_kernels[i] - 1);
                float[] cached_conv2_item = new float[cached_conv2_size];
                cached_conv2.Add(cached_conv2_item);
            }
            List<List<float[]>> statesList = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };
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
            for (int i = 0; i < 35; i++)
            {
                zzz.Add(stateList[i]);
                if ((i + 1) % 5 == 0)
                {
                    yyy.Add(zzz);
                    zzz = new List<float[]>();
                }
            }
            xxx.Add(yyy);
            return xxx;
        }

        public List<List<float[]>> stack_states(List<List<List<float[]>>> stateList)
        {
            int batch_size = stateList.Count;
            Debug.Assert(stateList[0].Count % 7 == 0, "when stack_states, state_list[0] is 7x");
            int num_encoders = stateList[0][0].Count;

            List<float[]> cached_len = new List<float[]>();
            List<float[]> cached_avg = new List<float[]>();
            List<float[]> cached_key = new List<float[]>();
            List<float[]> cached_val = new List<float[]>();
            List<float[]> cached_val2 = new List<float[]>();
            List<float[]> cached_conv1 = new List<float[]>();
            List<float[]> cached_conv2 = new List<float[]>();
            //cached_len
            List<List<float[]>> len_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                len_list.Add(stateList[n][0]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] len_avg = new float[len_list[0][i].Length * batch_size];
                int len_avg_item_length = len_list[0][i].Length;
                for (int x = 0; x < len_avg_item_length; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] len_avg_item = len_list[n][i];
                        Array.Copy(len_avg_item, x, len_avg, x * batch_size + n, 1);
                    }
                }
                cached_len.Add(len_avg);
            }
            //cached_avg
            List<List<float[]>> avg_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                avg_list.Add(stateList[n][1]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] avg = new float[avg_list[0][i].Length * batch_size];
                int avg_item_length = avg_list[0][i].Length;
                int cached_avg_axisnum = _customMetadata.Encoder_dims[i];
                for (int x = 0; x < avg_item_length/ cached_avg_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] avg_item = avg_list[n][i];
                        Array.Copy(avg_item, x* cached_avg_axisnum, avg, (x * batch_size + n)* cached_avg_axisnum, cached_avg_axisnum);
                        //Array.Copy(avg_item, n * avg_item_length + x, avg, x * batch_size + n, 1);
                    }
                }
                cached_avg.Add(avg);
            }

            //cached_key
            List<List<float[]>> key_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                key_list.Add(stateList[n][2]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] key = new float[key_list[0][i].Length * batch_size];
                int key_item_length = key_list[0][i].Length;
                int cached_key_axisnum = _customMetadata.Attention_dims[i];
                for (int x = 0; x < key_item_length / cached_key_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] key_item = key_list[n][i];
                        Array.Copy(key_item, x * cached_key_axisnum, key, (x * batch_size + n) * cached_key_axisnum, cached_key_axisnum);
                    }
                }
                cached_key.Add(key);
            }

            //cached_val
            List<List<float[]>> val_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                val_list.Add(stateList[n][3]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] val = new float[val_list[0][i].Length * batch_size];
                int val_item_length = val_list[0][i].Length;
                int cached_val_axisnum = _customMetadata.Attention_dims[i] / 2;
                for (int x = 0; x < val_item_length / cached_val_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] val_item = val_list[n][i];
                        Array.Copy(val_item, x * cached_val_axisnum, val, (x * batch_size + n) * cached_val_axisnum, cached_val_axisnum);
                    }
                }
                cached_val.Add(val);
            }

            //cached_val2
            List<List<float[]>> val2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                val2_list.Add(stateList[n][4]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] val2 = new float[val2_list[0][i].Length * batch_size];
                int val2_item_length = val2_list[0][i].Length;
                int cached_val2_axisnum = _customMetadata.Attention_dims[i] / 2;
                for (int x = 0; x < val2_item_length / cached_val2_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] val2_item = val2_list[n][i];
                        Array.Copy(val2_item, x * cached_val2_axisnum, val2, (x * batch_size + n) * cached_val2_axisnum, cached_val2_axisnum);
                    }
                }
                cached_val2.Add(val2);
            }

            //cached_conv1
            List<List<float[]>> conv1_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                conv1_list.Add(stateList[n][5]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] conv1 = new float[conv1_list[0][i].Length * batch_size];
                int conv1_item_length = conv1_list[0][i].Length;
                int cached_conv1_axisnum = _customMetadata.Encoder_dims[i] * (_customMetadata.Cnn_module_kernels[i] - 1);
                for (int x = 0; x < conv1_item_length / cached_conv1_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] conv1_item = conv1_list[n][i];
                        Array.Copy(conv1_item, x * cached_conv1_axisnum, conv1, (x * batch_size + n) * cached_conv1_axisnum, cached_conv1_axisnum);
                    }
                }
                cached_conv1.Add(conv1);
            }

            //cached_conv2
            List<List<float[]>> conv2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                conv2_list.Add(stateList[n][6]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] conv2 = new float[conv2_list[0][i].Length * batch_size];
                int conv2_item_length = conv2_list[0][i].Length;
                int cached_conv2_axisnum = _customMetadata.Encoder_dims[i] * (_customMetadata.Cnn_module_kernels[i] - 1);
                for (int x = 0; x < conv2_item_length / cached_conv2_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] conv2_item = conv2_list[n][i];
                        Array.Copy(conv2_item, x * cached_conv2_axisnum, conv2, (x * batch_size + n) * cached_conv2_axisnum, cached_conv2_axisnum);
                    }
                }
                cached_conv2.Add(conv2);
            }

            List<List<float[]>> states = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };

            return states;
        }
        public List<List<List<float[]>>> unstack_states(List<float[]> encoder_out_states)
        {
            List<List<List<float[]>>> statesList = new List<List<List<float[]>>>();
            Debug.Assert(encoder_out_states.Count % 7 == 0, "when stack_states, encoder_out_states[0] is 7x");
            int batch_size = encoder_out_states[0].Length / _customMetadata.Num_encoder_layers[0];
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            for (int i = 0; i < batch_size; i++)
            {
                List<float[]> cached_len = new List<float[]>();
                List<float[]> cached_avg = new List<float[]>();
                List<float[]> cached_key = new List<float[]>();
                List<float[]> cached_val = new List<float[]>();
                List<float[]> cached_val2 = new List<float[]>();
                List<float[]> cached_conv1 = new List<float[]>();
                List<float[]> cached_conv2 = new List<float[]>();
                for (int j = 0; j < 7; j++)
                {
                    for (int m = 0; m < num_encoders; m++)
                    {
                        int num_encoder_layers = _customMetadata.Num_encoder_layers[m];
                        float[] item = encoder_out_states[j * num_encoders + m];
                        int n = 1;
                        switch (j)
                        {
                            case 0:
                                int cached_len_size = num_encoder_layers * n;
                                float[] cached_len_item = new float[cached_len_size];
                                for (int k = 0; k < cached_len_size; k++)
                                {
                                    Array.Copy(item, item.Length / cached_len_size * k + i, cached_len_item, k, 1);
                                }
                                cached_len.Add(cached_len_item);
                                break;
                            case 1:
                                int cached_avg_axisnum = _customMetadata.Encoder_dims[m];
                                int cached_avg_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m];
                                float[] cached_avg_item = new float[cached_avg_size];
                                for (int k = 0; k < cached_avg_size/ cached_avg_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_avg_size * k + i)* cached_avg_axisnum, cached_avg_item, k* cached_avg_axisnum, cached_avg_axisnum);
                                }
                                cached_avg.Add(cached_avg_item);
                                break;
                            case 2:
                                int cached_key_axisnum = _customMetadata.Attention_dims[m];
                                int cached_key_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * _customMetadata.Attention_dims[m];
                                float[] cached_key_item = new float[cached_key_size];
                                for (int k = 0; k < cached_key_size / cached_key_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_key_size * k + i) * cached_key_axisnum, cached_key_item, k * cached_key_axisnum, cached_key_axisnum);
                                }
                                cached_key.Add(cached_key_item);
                                break;
                            case 3:
                                int cached_val_axisnum = _customMetadata.Attention_dims[m] / 2;
                                int cached_val_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * (_customMetadata.Attention_dims[m] / 2);
                                float[] cached_val_item = new float[cached_val_size];
                                for (int k = 0; k < cached_val_size / cached_val_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_val_size * k + i) * cached_val_axisnum, cached_val_item, k * cached_val_axisnum, cached_val_axisnum);
                                }
                                cached_val.Add(cached_val_item);
                                break;
                            case 4:
                                int cached_val2_axisnum = _customMetadata.Attention_dims[m] / 2;
                                int cached_val2_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * (_customMetadata.Attention_dims[m] / 2);
                                float[] cached_val2_item = new float[cached_val2_size];
                                for (int k = 0; k < cached_val2_size / cached_val2_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_val2_size * k + i) * cached_val2_axisnum, cached_val2_item, k * cached_val2_axisnum, cached_val2_axisnum);
                                }
                                cached_val2.Add(cached_val2_item);
                                break;
                            case 5:
                                int cached_conv1_axisnum = _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                int cached_conv1_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                float[] cached_conv1_item = new float[cached_conv1_size];
                                for (int k = 0; k < cached_conv1_size / cached_conv1_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_conv1_size * k + i) * cached_conv1_axisnum, cached_conv1_item, k * cached_conv1_axisnum, cached_conv1_axisnum);
                                }
                                cached_conv1.Add(cached_conv1_item);
                                break;
                            case 6:
                                int cached_conv2_axisnum = _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                int cached_conv2_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                float[] cached_conv2_item = new float[cached_conv2_size];
                                for (int k = 0; k < cached_conv2_size / cached_conv2_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_conv2_size * k + i) * cached_conv2_axisnum, cached_conv2_item, k * cached_conv2_axisnum, cached_conv2_axisnum);
                                }
                                cached_conv2.Add(cached_conv2_item);
                                break;

                        }
                    }
                }
                List<List<float[]>> states = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };
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
                List<float[]> items = statesList[i];
                for (int m = 0; m < items.Count; m++)
                {
                    var state = items[m];
                    string name = "";
                    int[] dim = new int[1];
                    switch (i)
                    {
                        case 0:
                            name = "cached_len";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], batchSize };
                            name = name + "_" + m.ToString();
                            Int64[] state2 = state.Select(x => Convert.ToInt64(x.ToString())).ToArray();
                            var tensor_len = new DenseTensor<Int64>(state2, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor_len));
                            break;
                        case 1:
                            name = "cached_avg";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], batchSize, onlineCustomMetadata.Encoder_dims[m] };
                            name = name + "_" + m.ToString();
                            var tensor_avg = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_avg));
                            break;
                        case 2:
                            name = "cached_key";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] };
                            name = name + "_" + m.ToString();
                            var tensor_key = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_key));
                            break;
                        case 3:
                            name = "cached_val";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] / 2 };
                            name = name + "_" + m.ToString();
                            var tensor_val = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val));
                            break;
                        case 4:
                            name = "cached_val2";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] / 2 };
                            name = name + "_" + m.ToString();
                            var tensor_val2 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val2));
                            break;
                        case 5:
                            name = "cached_conv1";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m],
                                batchSize, onlineCustomMetadata.Encoder_dims[m], onlineCustomMetadata.Cnn_module_kernels[m]-1};
                            name = name + "_" + m.ToString();
                            var tensor_conv1 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv1));
                            break;
                        case 6:
                            name = "cached_conv2";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m],
                                batchSize, onlineCustomMetadata.Encoder_dims[m], onlineCustomMetadata.Cnn_module_kernels[m]-1};
                            name = name + "_" + m.ToString();
                            var tensor_conv2 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv2));
                            break;
                    }
                }
            }
            try
            {
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
        ~OnlineProjOfZipformer()
        {
            Dispose(_disposed);
        }
    }
}
