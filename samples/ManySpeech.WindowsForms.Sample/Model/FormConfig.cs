using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.WindowsForms.Sample.Model
{
    internal class FormConfig
    {
        // Target child form type (required)
        public Type FormType { get; set; }

        // Parameters to pass (flexible definition: single parameter/multiple parameters/custom object)
        // Single string parameter (e.g., configuration path)
        public string StrParam { get; set; }

        // Multiple parameters of different types (use object[] to support any type)
        public object[] MultiParams { get; set; }

        // Custom complex parameter (type-safe)
        public RecognizerFormParam RecognizerFormParam { get; set; }
    }
    public class RecognizerFormParam
    {
        public RecognizerCategory RecognizerCategory { get; set; }
        public string ModelName { get; set; }
    }
}
