using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.SileroVad.Model
{
    public class VadResultEntity
    {
        private List<SegmentEntity> _segments = new List<SegmentEntity>();
        private List<int[]> _segment = new List<int[]>();
        private List<float[]> _waveforms = new List<float[]>();

        public List<int[]> Segment { get => _segment; set => _segment = value; }
        public List<float[]> Waveforms { get => _waveforms; set => _waveforms = value; }
        public List<SegmentEntity> Segments { get => _segments; set => _segments = value; }
    }
}
