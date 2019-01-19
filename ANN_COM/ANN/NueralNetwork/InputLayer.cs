using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NueralNetwork
{
    public class InputLayer:Layer
    {
        public InputLayer(double[,] Input, int _MiniBatchSize) : base(Input, _MiniBatchSize)
        {

        }
        public InputLayer(double[,,] Input, int _MiniBatchSize) : base(Input, _MiniBatchSize)
        {

        }
    }
}
