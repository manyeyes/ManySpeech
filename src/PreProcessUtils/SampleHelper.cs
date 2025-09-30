using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PreProcessUtils
{
    public class SampleHelper
    {
        public static string[] GetPaths(string modelBasePath, string modelName)
        {
            string[]? mediaFilePaths = Array.Empty<string>();
            if (!string.IsNullOrEmpty(modelBasePath) && !string.IsNullOrEmpty(modelName))
            {
                string fullPath = Path.Combine(modelBasePath, modelName);
                if (Directory.Exists(fullPath))
                {
                    mediaFilePaths = Directory.GetFiles(
                        path: fullPath,
                        searchPattern: "*.wav",
                        searchOption: SearchOption.AllDirectories
                    );
                }// 路径不正确时返回空数组
            }
            return mediaFilePaths;
        }
        /// <summary>
        ///  Get sample form file 
        /// </summary>
        /// <param name="modelBasePath"></param>
        /// <param name="modelName"></param>
        /// <param name="total_duration"></param>
        /// <param name="mediaFilePaths"></param>
        /// <param name="startIndex"></param>
        /// <param name="batchSize"></param>
        /// <returns>Tuple<List<float[]>, List<string>></returns>
        public static (List<float[]> sampleList, List<string> pathList)? GetSampleFormFile(string[]? mediaFilePaths, ref TimeSpan total_duration, int startIndex = 0, int batchSize = 0)
        {
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                return null;
            }
            List<float[]>? sampleList = new List<float[]>();
            List<string> pathList = new List<string>();
            if (batchSize == 0) { batchSize = mediaFilePaths.Length; }
            int n = 0;
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (n < startIndex)
                {
                    continue;
                }
                if (batchSize <= n - startIndex)
                {
                    break;
                }
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    float[]? sample = AudioHelper.GetFileSample(wavFilePath: mediaFilePath, duration: ref duration);
                    if (sample != null)
                    {
                        pathList.Add(mediaFilePath);
                        sampleList.Add(sample);
                        total_duration += duration;
                    }
                }
                n++;
            }
            return (sampleList, pathList);
        }
        /// <summary>
        /// Get chunk sample form file
        /// </summary>
        /// <param name="modelBasePath"></param>
        /// <param name="modelName"></param>
        /// <param name="total_duration"></param>
        /// <param name="mediaFilePaths"></param>
        /// <param name="startIndex"></param>
        /// <param name="batchSize"></param>
        /// <returns>(List<List<float[]>> samplesList, List<string> pathList)?</returns>
        public static (List<List<float[]>> samplesList, List<string> pathList)? GetChunkSampleFormFile(string[]? mediaFilePaths, ref TimeSpan total_duration, int startIndex = 0, int batchSize = 0, int chunkSize = 9600, int tailLength = 6)
        {
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                return null;
            }
            List<List<float[]>> samplesList = new List<List<float[]>>();
            List<string> pathList = new List<string>();
            List<float[]>? samples = new List<float[]>();
            if (batchSize == 0) { batchSize = mediaFilePaths.Length; }
            int n = 0;
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (n < startIndex)
                {
                    continue;
                }
                if (batchSize <= n - startIndex)
                {
                    break;
                }
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(mediaFilePath, ref duration, chunkSize);
                    if (samples.Count > 0)
                    {
                        for (int j = 0; j < tailLength; j++)
                        {
                            samples.Add(new float[chunkSize]);
                        }
                        pathList.Add(mediaFilePath);
                        samplesList.Add(samples);
                        total_duration += duration;
                    }
                }
                n++;
            }
            return (samplesList, pathList);
        }
    }
}
