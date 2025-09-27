using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AudioInOut.Base
{
    public abstract class BaseRecorder: IRecorder,IDisposable
    {
        public bool IsCapturing { get; private set; }
        public int SampleRate => 16000;
        public int BitsPerSample => 16;
        public int Channels => 1;

        public abstract void StartCapture();
        public abstract void StopCapture();
        public abstract Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken);

        public virtual void Dispose()
        {
            StopCapture();
        }

    }
}
