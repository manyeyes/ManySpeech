// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using ManySpeech.K2TransducerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.K2TransducerAsr
{
    internal class OnlineProjOfZipformer2ctc : IOnlineProj, IDisposable
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
        private int _sampleRate=16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public OnlineProjOfZipformer2ctc(OnlineModel onlineModel)
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

        public float[] GetEncoderInitProcessedLens(int batchSize = 1)
        {
            float[] processedLens = new float[batchSize];
            return processedLens;
        }
        public float[] GetEncoderInitEmbedStates(int batchSize = 1)
        {
            float[] embedStates = new float[batchSize * 128 * 3 * 19];
            return embedStates;
        }
        public List<List<float[]>> GetEncoderInitStates(int batchSize = 1)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            //计算尺寸
            for (int i = 0; i < num_encoders; i++)
            {
                int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                int embed_dim = _customMetadata.Encoder_dims[i];
                int ds = 1;// _customMetadata.Downsampling_factor[i];
                int num_heads = _customMetadata.Num_heads[i];
                int key_dim = _customMetadata.Query_head_dims[i] * num_heads;
                int value_dim = _customMetadata.Value_head_dims[i] * num_heads;
                int downsample_left = _customMetadata.Left_context_len[i] / ds;
                int nonlin_attn_head_dim = 3 * embed_dim / 4;
                int conv_left_pad = _customMetadata.Cnn_module_kernels[i] / 2;
                for (int layer = 0; layer < num_encoder_layers; layer++)
                {
                    //cached_key
                    int cached_key_size = downsample_left * batchSize * key_dim;
                    float[] cached_key_item = new float[cached_key_size];
                    //cached_nonlin_attn
                    int cached_nonlin_attn_size = 1 * batchSize * downsample_left * nonlin_attn_head_dim;
                    float[] cached_nonlin_attn_item = new float[cached_nonlin_attn_size];
                    //cached_val1
                    int cached_val1_size = downsample_left * batchSize * value_dim;
                    float[] cached_val1_item = new float[cached_val1_size];
                    //cached_val2
                    int cached_val2_size = downsample_left * batchSize * value_dim;
                    float[] cached_val2_item = new float[cached_val2_size];
                    //cached_conv1
                    int cached_conv1_size = batchSize * embed_dim * conv_left_pad;
                    float[] cached_conv1_item = new float[cached_conv1_size];
                    //cached_conv2
                    int cached_conv2_size = batchSize * embed_dim * conv_left_pad;
                    float[] cached_conv2_item = new float[cached_conv2_size];
                    List<float[]> states = new List<float[]>();
                    states = new List<float[]> { cached_key_item, cached_nonlin_attn_item, cached_val1_item, cached_val2_item, cached_conv1_item, cached_conv2_item };
                    statesList.Add(states);
                }
            }
            float[] embed_states_item = GetEncoderInitEmbedStates(batchSize);
            List<float[]> embed_states_list = new List<float[]> { embed_states_item };
            statesList.Add(embed_states_list);
            float[] processed_lens_item = GetEncoderInitProcessedLens(batchSize);
            List<float[]> processed_lens_list = new List<float[]> { processed_lens_item };
            statesList.Add(processed_lens_list);
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
            for (int i = 0; i < 96; i++)
            {
                zzz.Add(stateList[i]);
                if ((i + 1) % 6 == 0)
                {
                    yyy.Add(zzz);
                    zzz = new List<float[]>();
                }
            }
            zzz = new List<float[]>();
            zzz.Add(stateList[96]);
            yyy.Add(zzz);
            zzz = new List<float[]>();
            zzz.Add(stateList[97]);
            yyy.Add(zzz);
            xxx.Add(yyy);
            return xxx;
        }

        public List<List<float[]>> stack_states(List<List<List<float[]>>> stateList)
        {

            int batch_size = stateList.Count;
            int dim = _customMetadata.Num_encoder_layers.Sum(x=>x);
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            List<float[]> cached_key = new List<float[]>();
            List<float[]> cached_nonlin_attn = new List<float[]>();
            List<float[]> cached_val1 = new List<float[]>();
            List<float[]> cached_val2 = new List<float[]>();
            List<float[]> cached_conv1 = new List<float[]>();
            List<float[]> cached_conv2 = new List<float[]>();
            //cached_key
            List<List<float[]>> key_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][0]);
                }
                key_list.Add(cache);
            }
            //cached_nonlin_attn
            List<List<float[]>> nonlin_attn_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][1]);
                }
                nonlin_attn_list.Add(cache);
            }
            //cached_val1
            List<List<float[]>> val1_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][2]);
                }
                val1_list.Add(cache);
            }
            //cached_val2
            List<List<float[]>> val2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][3]);
                }
                val2_list.Add(cache);
            }
            //cached_conv1
            List<List<float[]>> conv1_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][4]);
                }
                conv1_list.Add(cache);
            }
            //cached_conv2
            List<List<float[]>> conv2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                List<float[]> cache = new List<float[]>();
                for (int i = 0; i < dim; i++)
                {
                    cache.Add(stateList[n][i][5]);
                }
                conv2_list.Add(cache);
            }
            int y = 0;
            for (int i = 0; i < num_encoders; i++)
            {
                int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                int embed_dim = _customMetadata.Encoder_dims[i];
                int ds = 1;//_customMetadata.Downsampling_factor[i];
                int num_heads = _customMetadata.Num_heads[i];
                int key_dim = _customMetadata.Query_head_dims[i] * num_heads;
                int value_dim = _customMetadata.Value_head_dims[i] * num_heads;
                int downsample_left = _customMetadata.Left_context_len[i] / ds;
                int nonlin_attn_head_dim = 3 * embed_dim / 4;
                int conv_left_pad = _customMetadata.Cnn_module_kernels[i] / 2;
                for (int layer = 0; layer < num_encoder_layers; layer++)
                {
                    //cached_key stack
                    float[] key = new float[key_list[0][y].Length * batch_size];
                    int key_item_length = key_list[0][y].Length;
                    int cached_key_axisnum = key_dim;
                    for (int x = 0; x < key_item_length / cached_key_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] key_item = key_list[n][y];
                            Array.Copy(key_item, x * cached_key_axisnum, key, (x * batch_size + n) * cached_key_axisnum, cached_key_axisnum);
                        }
                    }
                    cached_key.Add(key);
                    //cached_nonlin_attn stack
                    float[] nonlin_attn = new float[nonlin_attn_list[0][y].Length * batch_size];
                    int nonlin_attn_item_length = nonlin_attn_list[0][y].Length;
                    int cached_nonlin_attn_axisnum = downsample_left;
                    for (int x = 0; x < nonlin_attn_item_length / cached_nonlin_attn_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] nonlin_attn_item = nonlin_attn_list[n][y];
                            Array.Copy(nonlin_attn_item, x * cached_nonlin_attn_axisnum, nonlin_attn, (x * batch_size + n) * cached_nonlin_attn_axisnum, cached_nonlin_attn_axisnum);
                        }
                    }
                    cached_nonlin_attn.Add(nonlin_attn);
                    //cached_val1 stack
                    float[] val1 = new float[val1_list[0][y].Length * batch_size];
                    int val1_item_length = val1_list[0][y].Length;
                    int cached_val1_axisnum = value_dim;
                    for (int x = 0; x < val1_item_length / cached_val1_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] val1_item = val1_list[n][y];
                            Array.Copy(val1_item, x * cached_val1_axisnum, val1, (x * batch_size + n) * cached_val1_axisnum, cached_val1_axisnum);
                        }
                    }
                    cached_val1.Add(val1);
                    //cached_val2 stack
                    float[] val2 = new float[val2_list[0][y].Length * batch_size];
                    int val2_item_length = val2_list[0][y].Length;
                    int cached_val2_axisnum = value_dim;
                    for (int x = 0; x < val2_item_length / cached_val2_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] val2_item = val2_list[n][y];
                            Array.Copy(val2_item, x * cached_val2_axisnum, val2, (x * batch_size + n) * cached_val2_axisnum, cached_val2_axisnum);
                        }
                    }
                    cached_val2.Add(val2);
                    //cached_conv1 stack
                    float[] conv1 = new float[conv1_list[0][y].Length * batch_size];
                    int conv1_item_length = conv1_list[0][y].Length;
                    int cached_conv1_axisnum = embed_dim * conv_left_pad;
                    for (int x = 0; x < conv1_item_length / cached_conv1_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] conv1_item = conv1_list[n][y];
                            Array.Copy(conv1_item, x * cached_conv1_axisnum, conv1, (x * batch_size + n) * cached_conv1_axisnum, cached_conv1_axisnum);
                        }
                    }
                    cached_conv1.Add(conv1);
                    //cached_conv2 stack
                    float[] conv2 = new float[conv2_list[0][y].Length * batch_size];
                    int conv2_item_length = conv2_list[0][y].Length;
                    int cached_conv2_axisnum = embed_dim * conv_left_pad;
                    for (int x = 0; x < conv2_item_length / cached_conv2_axisnum; x++)
                    {
                        for (int n = 0; n < batch_size; n++)
                        {
                            float[] conv2_item = conv2_list[n][y];
                            Array.Copy(conv2_item, x * cached_conv2_axisnum, conv2, (x * batch_size + n) * cached_conv2_axisnum, cached_conv2_axisnum);
                        }
                    }
                    cached_conv2.Add(conv2);
                    y++;
                }
            }
            //embed_states
            List<List<float[]>> embed_states_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                embed_states_list.Add(stateList[n][dim]);
            }
            //stack
            float[] embed_states = new float[embed_states_list[0][0].Length * batch_size];
            int embed_states_item_length = embed_states_list[0][0].Length;
            int embed_states_axisnum = 128 * 3 * 19;
            for (int x = 0; x < embed_states_item_length / embed_states_axisnum; x++)
            {
                for (int n = 0; n < batch_size; n++)
                {
                    float[] embed_states_item = embed_states_list[n][0];
                    Array.Copy(embed_states_item, x * embed_states_axisnum, embed_states, (x * batch_size + n) * embed_states_axisnum, embed_states_axisnum);
                }
            }
            List<float[]> cache_embed_states = new List<float[]>();
            cache_embed_states.Add(embed_states);
            //processed_lens
            List<List<float[]>> processed_lens_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                processed_lens_list.Add(stateList[n][dim+1]);
            }
            //stack
            float[] processed_lens = new float[processed_lens_list[0][0].Length * batch_size];
            int processed_lens_item_length = processed_lens_list[0][0].Length;
            int processed_lens_axisnum = 1;
            for (int x = 0; x < processed_lens_item_length / processed_lens_axisnum; x++)
            {
                for (int n = 0; n < batch_size; n++)
                {
                    float[] processed_lens_item = processed_lens_list[n][0];
                    Array.Copy(processed_lens_item, x * processed_lens_axisnum, processed_lens, (x * batch_size + n) * processed_lens_axisnum, processed_lens_axisnum);
                }
            }
            List<float[]> cache_processed_lens = new List<float[]>();
            cache_processed_lens.Add(processed_lens);
            List<List<float[]>> states = new List<List<float[]>> { cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2, cache_embed_states, cache_processed_lens };

            return states;
        }
        public List<List<List<float[]>>> unstack_states(List<float[]> encoder_out_states)
        {
            int dim = _customMetadata.Num_encoder_layers.Sum(x => x);
            List<List<List<float[]>>> statesList = new List<List<List<float[]>>>();
            int batch_size = encoder_out_states[0].Length / (_customMetadata.Left_context_len[0] * _customMetadata.Query_head_dims[0] * _customMetadata.Num_heads[0]);
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            for (int i = 0; i < batch_size; i++)
            {
                List<float[]> embed_states = new List<float[]>();
                List<float[]> processed_lens = new List<float[]>();

                List<List<float[]>> states = new List<List<float[]>>();
                int y = 0;//encoder_out_states layer
                int n = 1;//batch size
                for (int j = 0; j < num_encoders; j++)
                {
                    int num_encoder_layers = _customMetadata.Num_encoder_layers[j];
                    int embed_dim = _customMetadata.Encoder_dims[j];
                    int ds = 1;//_customMetadata.Downsampling_factor[j];
                    int num_heads = _customMetadata.Num_heads[j];
                    int key_dim = _customMetadata.Query_head_dims[j] * num_heads;
                    int value_dim = _customMetadata.Value_head_dims[j] * num_heads;
                    int downsample_left = _customMetadata.Left_context_len[j] / ds;
                    int nonlin_attn_head_dim = 3 * embed_dim / 4;
                    int conv_left_pad = _customMetadata.Cnn_module_kernels[j] / 2;
                    for (int layer = 0; layer < num_encoder_layers; layer++)
                    {
                        List<float[]> statesItem = new List<float[]>();
                        for (int m = 0; m < 6; m++)
                        {
                            float[] item = encoder_out_states[y * 6 + m];
                            switch (m)
                            {
                                case 0:
                                    int cached_key_axisnum = key_dim;
                                    int cached_key_size = downsample_left * n * key_dim;
                                    float[] cached_key_item = new float[cached_key_size];
                                    for (int k = 0; k < cached_key_size / cached_key_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_key_size * k + i) * cached_key_axisnum, cached_key_item, k * cached_key_axisnum, cached_key_axisnum);
                                    }
                                    statesItem.Add(cached_key_item);
                                    break;
                                case 1:
                                    int cached_nonlin_attn_axisnum = downsample_left * nonlin_attn_head_dim;
                                    int cached_nonlin_attn_size = 1 * n * downsample_left * nonlin_attn_head_dim;
                                    float[] cached_nonlin_attn_item = new float[cached_nonlin_attn_size];
                                    for (int k = 0; k < cached_nonlin_attn_size / cached_nonlin_attn_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_nonlin_attn_size * k + i) * cached_nonlin_attn_axisnum, cached_nonlin_attn_item, k * cached_nonlin_attn_axisnum, cached_nonlin_attn_axisnum);
                                    }
                                    statesItem.Add(cached_nonlin_attn_item);
                                    break;
                                case 2:
                                    int cached_val1_axisnum = value_dim;
                                    int cached_val1_size = downsample_left * n * value_dim;
                                    float[] cached_val1_item = new float[cached_val1_size];
                                    for (int k = 0; k < cached_val1_size / cached_val1_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_val1_size * k + i) * cached_val1_axisnum, cached_val1_item, k * cached_val1_axisnum, cached_val1_axisnum);
                                    }
                                    statesItem.Add(cached_val1_item);
                                    break;
                                case 3:
                                    int cached_val2_axisnum = value_dim;
                                    int cached_val2_size = downsample_left * n * value_dim;
                                    float[] cached_val2_item = new float[cached_val2_size];
                                    for (int k = 0; k < cached_val2_size / cached_val2_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_val2_size * k + i) * cached_val2_axisnum, cached_val2_item, k * cached_val2_axisnum, cached_val2_axisnum);
                                    }
                                    statesItem.Add(cached_val2_item);
                                    break;
                                case 4:
                                    int cached_conv1_axisnum = embed_dim * conv_left_pad;
                                    int cached_conv1_size = n * embed_dim * conv_left_pad;
                                    float[] cached_conv1_item = new float[cached_conv1_size];
                                    for (int k = 0; k < cached_conv1_size / cached_conv1_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_conv1_size * k + i) * cached_conv1_axisnum, cached_conv1_item, k * cached_conv1_axisnum, cached_conv1_axisnum);
                                    }
                                    statesItem.Add(cached_conv1_item);
                                    break;
                                case 5:
                                    int cached_conv2_axisnum = embed_dim * conv_left_pad;
                                    int cached_conv2_size = n * embed_dim * conv_left_pad;
                                    float[] cached_conv2_item = new float[cached_conv2_size];
                                    for (int k = 0; k < cached_conv2_size / cached_conv2_axisnum; k++)
                                    {
                                        Array.Copy(item, (item.Length / cached_conv2_size * k + i) * cached_conv2_axisnum, cached_conv2_item, k * cached_conv2_axisnum, cached_conv2_axisnum);
                                    }
                                    statesItem.Add(cached_conv2_item);
                                    break;

                            }
                        }
                        states.Add(statesItem);
                        y++;
                    }

                }
                float[] encoder_out_embed_states = encoder_out_states[dim*6];
                int embed_states_axisnum = 128 * 3 * 19;
                int embed_states_size = n * 128 * 3 * 19;
                float[] embed_states_item = new float[embed_states_size];
                for (int k = 0; k < embed_states_size / embed_states_axisnum; k++)
                {
                    Array.Copy(encoder_out_embed_states, (encoder_out_embed_states.Length / embed_states_size * k + i) * embed_states_axisnum, embed_states_item, k * embed_states_axisnum, embed_states_axisnum);
                }
                embed_states.Add(embed_states_item);
                states.Add(embed_states);

                float[] encoder_out_processed_lens = encoder_out_states[dim*6+1];
                int processed_lens_axisnum = 1;
                int processed_lens_size = n * 1;
                float[] processed_lens_item = new float[processed_lens_size];
                for (int k = 0; k < processed_lens_size / processed_lens_axisnum; k++)
                {
                    Array.Copy(encoder_out_processed_lens, (encoder_out_processed_lens.Length / processed_lens_size * k + i) * processed_lens_axisnum, processed_lens_item, k * processed_lens_axisnum, processed_lens_axisnum);
                }
                processed_lens.Add(processed_lens_item);
                states.Add(processed_lens);
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
                int num_encoders = _customMetadata.Num_encoder_layers.Length;
                for (int m = 0; m < statesList.Count - 2; m++)
                {
                    List<float[]> items = statesList[m];
                    int y = 0;
                    for (int i = 0; i < num_encoders; i++)
                    {
                        int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                        int embed_dim = _customMetadata.Encoder_dims[i];
                        int ds = 1;//_customMetadata.Downsampling_factor[i];
                        int num_heads = _customMetadata.Num_heads[i];
                        int key_dim = _customMetadata.Query_head_dims[i] * num_heads;
                        int value_dim = _customMetadata.Value_head_dims[i] * num_heads;
                        int downsample_left = _customMetadata.Left_context_len[i] / ds;
                        int nonlin_attn_head_dim = 3 * embed_dim / 4;
                        int conv_left_pad = _customMetadata.Cnn_module_kernels[i] / 2;
                        int[] dim = new int[1];
                        string name = "";
                        for (int layer = 0; layer < num_encoder_layers; layer++)
                        {
                            var state = items[y];
                            switch (m)
                            {
                                case 0:
                                    name = "cached_key";
                                    dim = new int[] { downsample_left, batchSize, key_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_len = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_len));
                                    break;
                                case 1:
                                    name = "cached_nonlin_attn";
                                    dim = new int[] { 1, batchSize, downsample_left, nonlin_attn_head_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_avg = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_avg));
                                    break;
                                case 2:
                                    name = "cached_val1";
                                    dim = new int[] { downsample_left, batchSize, value_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_val = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val));
                                    break;
                                case 3:
                                    name = "cached_val2";
                                    dim = new int[] { downsample_left, batchSize, value_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_val2 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val2));
                                    break;
                                case 4:
                                    name = "cached_conv1";
                                    dim = new int[] { batchSize, embed_dim, conv_left_pad };
                                    name = name + "_" + y.ToString();
                                    var tensor_conv1 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv1));
                                    break;
                                case 5:
                                    name = "cached_conv2";
                                    dim = new int[] { batchSize, embed_dim, conv_left_pad };
                                    name = name + "_" + y.ToString();
                                    var tensor_conv2 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv2));
                                    break;
                            }
                            y++;
                        }
                    }
                }
                int[] embed_states_dim = new int[] { batchSize, 128, 3, 19 };
                float[] embed_states_value = statesList[6][0];
                var embed_states = new DenseTensor<float>(embed_states_value, embed_states_dim, false);
                container.Add(NamedOnnxValue.CreateFromTensor<float>("embed_states", embed_states));
                int[] processed_lens_dim = new int[] { batchSize };
                float[] processed_lens_value = statesList[7][0];
                Int64[] processed_lens_value2 = processed_lens_value.Select(x => Convert.ToInt64(x.ToString())).ToArray();
                var processed_lens = new DenseTensor<Int64>(processed_lens_value2, processed_lens_dim, false);
                container.Add(NamedOnnxValue.CreateFromTensor<Int64>("processed_lens", processed_lens));

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
            return null;
        }

        public JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {            
            return null;
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
        ~OnlineProjOfZipformer2ctc()
        {
            Dispose(_disposed);
        }
    }
}
