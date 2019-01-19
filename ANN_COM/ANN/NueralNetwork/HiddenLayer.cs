using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANNMath;

namespace NueralNetwork
{
    public class HiddenLayer: Layer
    {
        public HiddenLayer(double[,] InputDims, int ThisLayersNeurons, int _MiniBatchSize, ActivationFunction _ActivationFunction, WeightInitialization _WeightInitialization, TrainAlgorithm _TrainAlgorithm, double _LearnRate, double _L2_Regularization, double _gradient) : base(InputDims, ThisLayersNeurons, _MiniBatchSize, _ActivationFunction, _WeightInitialization, _TrainAlgorithm, _LearnRate, _L2_Regularization, _gradient)
        {
           
        }   
        public override void BackPropagate(double[,] OutgoingWeights, double[,] NextLayersD)
        {
            double[,] OutgoingWeights_noBias = new double[OutgoingWeights.GetLength(0) - Bias, OutgoingWeights.GetLength(1)];//there's no Delta for BiasNeurons
            for (int i = 0; i < OutgoingWeights_noBias.GetLength(0); i++)
            {
                for (int j = 0; j < OutgoingWeights_noBias.GetLength(1); j++)
                {
                    OutgoingWeights_noBias[i, j] = OutgoingWeights[i, j];
                }
            }
            D = ANNMath.ANNMath.ElementwiseMultiplication(dF, ANNMath.ANNMath.MultiplyMatrices(OutgoingWeights_noBias, NextLayersD));
        }
    }
}
