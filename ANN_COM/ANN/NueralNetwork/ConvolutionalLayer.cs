using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANNMath;

namespace NueralNetwork
{
    public class ConvolutionalLayer: Layer
    {        
        public ConvolutionalLayer(double[,,] InputDims_ZofPreviousLayer, int FilterSize, int Stride, int Depth, int _MiniBatchSize, ActivationFunction _ActivationFunction, WeightInitialization _WeightInitialization, TrainAlgorithm _TrainAlgorithm, double _LearnRate, double _L2_Regularization, double _gradient) : base(InputDims_ZofPreviousLayer, FilterSize, Stride, Depth, _MiniBatchSize, _ActivationFunction, _WeightInitialization, _TrainAlgorithm, _LearnRate, _L2_Regularization, _gradient)
        {
            
        }
        public override void BackPropagate(double[,] OutgoingWeights, double[,] NextLayersD)// Backprop From FC-Layer to 1st COnvLayer
        {
            double[,] OutgoingWeights_noBias = new double[OutgoingWeights.GetLength(0) - Bias, OutgoingWeights.GetLength(1)];//there's no Delta for BiasNeurons
            for (int i = 0; i < OutgoingWeights_noBias.GetLength(0); i++)
            {
                for (int j = 0; j < OutgoingWeights_noBias.GetLength(1); j++)
                {
                    OutgoingWeights_noBias[i, j] = OutgoingWeights[i, j];
                }
            }
            double[,] D_temp = ANNMath.ANNMath.ElementwiseMultiplication(dF, ANNMath.ANNMath.MultiplyMatrices(OutgoingWeights_noBias, NextLayersD));//OutgoingWeights_noBias[Heigth*Width*Depth,NextLayersNeurons] || NextLayersD[NextLayersNeurons,MiniBatchsize]
            //D_temp[höhe*weite*tiefe,MiniBatchSize] formatieren zu D[höhe*weite*MiniBatchSize,teife]      
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int i = 0; i < Depth; i++)
                {
                    for (int j = 0; j < (Height * Width); j++)
                    {
                        D[j + b * ((Height * Width)), i] = D_temp[i * (Height * Width) + j, b];
                    }
                }
            }
            //D als D_3D Formatieren
            D_3D = ANNMath.ANNMath.ComputeD_3D(D, D_3D, MiniBatchSize);
        }
        public override void BackPropagate(double[,] NextLayersWeightsRotated, int NextLayersFilterSize, double[,] NextLayersD)// Backprop betweent ConvLayer
        {
            //NextLayersD is already aligned and zeropadding added
            double[,] NextLayersWeightsRotated_noBias = new double[NextLayersWeightsRotated.GetLength(0) - Bias, NextLayersWeightsRotated.GetLength(1)];//there's no Delta for BiasNeurons
            for (int i = 0; i < NextLayersWeightsRotated_noBias.GetLength(0); i++)
            {
                for (int j = 0; j < NextLayersWeightsRotated_noBias.GetLength(1); j++)
                {
                    NextLayersWeightsRotated_noBias[i, j] = NextLayersWeightsRotated[i, j];
                }
            }
            double[,] NextLayersW_unrot_NoBias = ANNMath.ANNMath.UnRotateWeightMat(NextLayersWeightsRotated_noBias, NextLayersFilterSize);//Format[NextLayersFilterSize*NextLayersFilterSize*NextLayersDepth, thisLayersDepth]
            //dF kommt als vektor vom Format [höhe*weite*tiefe,MiniBatchSize] soll aber eine Matrix vom Format [höhe*weite*MiniBatchSize, tiefe] sein, da ANNMath.ANNMath.MultiplyMatrices(NextLayersD, NextLayersW_unrot_NoBias) das Format[höhe*weite*MiniBatchSize, tiefe] besitzt
            double[,] dF_temp = new double[Height * Width * MiniBatchSize, Depth];
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int i = 0; i < Depth; i++)
                {
                    for (int j = 0; j < (Height * Width); j++)
                    {
                        dF_temp[j + b * (Height * Width), i] = dF[i * (Height * Width) + j, b];
                    }
                }
            }
            D = ANNMath.ANNMath.ElementwiseMultiplication(dF_temp, ANNMath.ANNMath.MultiplyMatrices(NextLayersD, NextLayersW_unrot_NoBias));
            //D als D_3D Formatieren
            D_3D = ANNMath.ANNMath.ComputeD_3D(D, D_3D, MiniBatchSize);
        }
    }
}
