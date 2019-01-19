using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NueralNetwork
{
    public class OutputLayer : Layer
    {
        public OutputLayer(double[,] InputDims, int ThisLayersNeurons, int _MiniBatchSize, ActivationFunction _ActivationFunction, WeightInitialization _WeightInitialization, TrainAlgorithm _TrainAlgorithm, double _LearnRate, double _L2_Regularization, double _gradient) : base(InputDims, ThisLayersNeurons, _MiniBatchSize, _ActivationFunction, _WeightInitialization, _TrainAlgorithm, _LearnRate, _L2_Regularization, _gradient)
        {
        }
        public override void BackPropagate(double[,] error)
        {
            D = ANNMath.ANNMath.ElementwiseMultiplication(dF, error);
        }
    }
}
