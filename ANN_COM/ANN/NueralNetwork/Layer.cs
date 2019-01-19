using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NueralNetwork
{
    public class Layer
    {
        //Z is the matrix that holds output values
        public double[,] Z;
        public double[,,] Z_3D = null;
        //W is the ingoing weight matrix for this layer
        public double[,] W;
        //safe the Gradient of the last step to calculate conjugent gradient
        public double[,] dW_t_minus_1 = null;
        //adaptive learnrate Gain for each of the weights
        public double[,] AdaptiveLearnRateGain;
        //velocity Matrix for Adam-Optimizer
        public double[,] Velocity_Adam;
        //momentum Matrix for Adam-Optimizer
        public double[,] Momentum_Adam;
        public double beta1_Adam = 0.9;
        public double beta2_Adam = 0.999;
        public double epsilon_Adam = 0.00000001;//to avoid divisioin by zero
        public Int64 timestep_Adam = 1;
        //PreInput is the Matrix that holds the zeropadded, Bias-added and arranged Input for the Computation of S
        public double[,] PreInput;  
        //S is the matrix that holds the inputs to this layer
        public double[,] S;
        //D is the matrix that holds the deltas for this layer
        public double[,] D;
        public double[,,] D_3D = null;
        //dF is the matrix that holds the derivatives of the activation function
        public double[,] dF;
        public ActivationFunction ActivationFunction;
        public TrainAlgorithm TrainAlgorithm;
        public Random rnd = new Random();
        double LearnRate;
        double gradient = 1;
        double lReluByPassFactor = 0.001;
        public int Bias = 1;
        public int FilterSize = 0;
        public int Stride = 0;
        public int Height;
        public int Width;
        public int Depth;
        public int MiniBatchSize;
        public double L2_Regularization;
        private double SoftMax_Sum_Exp;
        private bool first_update = true;
        private double beta = 0;//used for calculating conjugate gradient
        private double beta_denominator = 0;//used for calculating conjugate gradient


        public Layer(double[,] Input, int _MiniBatchSize)//Constructor For InputLayer for fullyconnectet Networks
        {
            MiniBatchSize = _MiniBatchSize;
            //Z is the matrix that holds the outputs to this layer
            Z = new double[Input.GetLength(0), Input.GetLength(1) + Bias];//[Rows=BatchSize,Columns=InputsPerBatch+Bias] 
            for (int i = 0; i < Input.GetLength(0); i++)
            {
                for (int j = 0; j < Input.GetLength(1); j++)
                {
                    Z[i, j] = Input[i,j];
                }
            }
            //Setting Bias of Z to 1
            for (int i = 0; i < Z.GetLength(0); i++)//Z.GetLenght(0) = BatchSize
            {
                Z[i, Z.GetLength(1) - Bias] = 1;
            }
        }
        public Layer(double[,,] Input, int _MiniBatchSize)//Constructor For InputLayer for Convolutional Networks
        {
            //Z is the matrix that holds the outputs to this layer
            //Bsp für BGRA Bild (höhexweite=31x31), 24 Bilder Pro Batch = Input[31,31,4*24]
            MiniBatchSize = _MiniBatchSize;
            Z_3D = Input;//Bias is Added In Im2Mat    
            Z = new double[MiniBatchSize, Input.GetLength(0) * Input.GetLength(1) * Input.GetLength(2) / MiniBatchSize + Bias];//[Row = MiniBatchSize, Column=Height *  Width * Depth + Bias]
            //Setting Bias of Z to 1
            for (int i = 0; i < Z.GetLength(0); i++)//Z.GetLenght(0) = BatchSize
            {
                Z[i, Z.GetLength(1) - Bias] = 1;
            }
            Z = ANNMath.ANNMath.ComputeZ(Z, Z_3D, MiniBatchSize); 
        }
        public Layer(double[,] InputDims_ZofPreviousLayer, int ThisLayersNeurons, int _MiniBatchSize, ActivationFunction _ActivationFunction, WeightInitialization _WeightInitialization, TrainAlgorithm _TrainAlgorithm, double _LearnRate, double _L2_Regularization, double _gradient)//Constructor For HiddenLayer and OutputLayer
        {
            TrainAlgorithm = _TrainAlgorithm;
            MiniBatchSize = _MiniBatchSize;
            gradient = _gradient;
            L2_Regularization = _L2_Regularization;
            LearnRate = _LearnRate;
            ActivationFunction = _ActivationFunction;
            //S is the matrix that holds the inputs to this layer
            S = new double[InputDims_ZofPreviousLayer.GetLength(0), ThisLayersNeurons];//[Rows=BatchSize,Columns] 
            //W is the ingoing weight matrix for this layer
            W = new double[InputDims_ZofPreviousLayer.GetLength(1), ThisLayersNeurons];//[Rows=PreviousLayersNeurons + Bias,Columns=ThisLayersNeurons]          
            if(_WeightInitialization == WeightInitialization.NormalizedGaussianRandom)
            {
                double mean = 0;
                double stdDev = 1;
                for (int i = 0; i < W.GetLength(0); i++)
                {
                    for (int j = 0; j < W.GetLength(1); j++)
                    {
                        double u1 = 1.0 - rnd.NextDouble(); //uniform(0,1] random doubles
                        double u2 = 1.0 - rnd.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                        W[i, j] = mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
                    }
                }
            }
            else if(_WeightInitialization == WeightInitialization.Random)
            {
                for (int i = 0; i < W.GetLength(0); i++)
                {
                    for (int j = 0; j < W.GetLength(1); j++)
                    {
                        W[i, j] = rnd.NextDouble() * 2 - 1;
                    }
                }
            }
            //safe previous gradient for adaptive learnrate
            dW_t_minus_1 = new double[W.GetLength(0), W.GetLength(1)];
            //initialize dW_t_minus_1 as zeros-Matrix
            for (int i = 0; i < dW_t_minus_1.GetLength(0); i++)
            {
                for (int j = 0; j < dW_t_minus_1.GetLength(1); j++)
                {
                    dW_t_minus_1[i, j] = 0;
                }
            }
            //initialaze the AdaptiveLearnRateGain
            AdaptiveLearnRateGain = new double[W.GetLength(0), W.GetLength(1)];
            //initialize AdaptiveLearnRateGain as ones-Matrix
            for (int i = 0; i < AdaptiveLearnRateGain.GetLength(0); i++)
            {
                for (int j = 0; j < AdaptiveLearnRateGain.GetLength(1); j++)
                {
                    AdaptiveLearnRateGain[i, j] = 1;
                }
            }
            if(TrainAlgorithm == TrainAlgorithm.Adam)
            {
                //initialize Velocity_Adam as zeros-Matrix
                Velocity_Adam = new double[W.GetLength(0), W.GetLength(1)];
                //initialize Momentum_Adam as zeros-Matrix
                Momentum_Adam = new double[W.GetLength(0), W.GetLength(1)];
            }
            //Z is the matrix that holds output values
            if (GetType() == typeof(HiddenLayer))
            {
                Z = new double[S.GetLength(0), S.GetLength(1) + Bias];//[Rows=BatchSize,Columns=ThisLayersNeurons + Bias]
                                                                                          //Setting Bias-Column (= Last Column) of Z to 1
                for (int i = 0; i < Z.GetLength(0); i++)//Z.GetLenght(0) = BatchSize
                {
                    Z[i, Z.GetLength(1) - Bias] = 1;
                }
            }
            else//if OutputLayer, there's no Bias on Z
            {
                this.Z = new double[this.S.GetLength(0), this.S.GetLength(1)];
            }
            //D is the matrix that holds the deltas for this layer
            this.D = new double[this.S.GetLength(1), this.S.GetLength(0)];//[Rows=ThisLayersNeurons,Columns=BatchSize]
            //dF is the matrix that holds the derivatives of the activation function. Bias doesent have a derevative
            this.dF = new double[this.S.GetLength(1), this.S.GetLength(0)];//[Rows=ThisLayersNeurons,Columns=BatchSize]
        }

        public Layer(double[,,] InputDims_ZofPreviousLayer, int _FilterSize, int _Stride, int _Depth, int _MiniBatchSize, ActivationFunction _ActivationFunction, WeightInitialization _WeightInitialization, TrainAlgorithm _TrainAlgorithm, double _LearnRate, double _L2_Regularization, double _gradient)//Constructor For ConvolutionalLayer
        {
            TrainAlgorithm = _TrainAlgorithm;
            MiniBatchSize = _MiniBatchSize;
            gradient = _gradient;
            L2_Regularization = _L2_Regularization;
            Stride = _Stride;
            //InputDims_ZofPreviousLayer.GetLength(0) = Height(i-1)
            //InputDims_ZofPreviousLayer.GetLength(1) =Width(i-1)
            //InputDims_ZofPreviousLayer.GetLength(2) = Depth(i-1)*MiniBacthSize = DepthOfInputLayer*MiniBatchSize
            double dHeight = (InputDims_ZofPreviousLayer.GetLength(0) - 1.0) / Stride + 1.0;//when changing something here, also change in ANNMAth.Im2Mat
            double dWidth = (InputDims_ZofPreviousLayer.GetLength(1) - 1.0) / Stride + 1.0;//when changing something here, also change in ANNMAth.Im2Mat
            if (dHeight % 1 == 0 && dWidth % 1 == 0)
            {
                Height = (int)dHeight;
                Width = (int)dWidth;
            }
            else
            {
                throw new Exception("xxxx");
            }  
            Depth = _Depth;
            FilterSize = _FilterSize;
            if (FilterSize % 2 == 0)//Filtersize must be an uneaven number
            {
                throw new Exception("xxxx");
            }
            Stride = _Stride;
            LearnRate = _LearnRate;
            ActivationFunction = _ActivationFunction;
            //PreInput is the Matrix that holds the zeropadded, Bias-added and arranged Input for the Computation of S
            PreInput = new double[Height * Width * MiniBatchSize, FilterSize * FilterSize * InputDims_ZofPreviousLayer.GetLength(2)/MiniBatchSize + Bias];
            //S is the matrix that holds the inputs to this layer
            S = new double[Height * Width * MiniBatchSize, Depth];
            //W is the ingoing rotated weight matrix for this layer inclusive a Bias for each Kernel
            W = new double[FilterSize * FilterSize * InputDims_ZofPreviousLayer.GetLength(2) / MiniBatchSize + Bias, Depth];//Have a look at page 11 in Theory, rotated WeightMatrix for ForewardPass
            if (_WeightInitialization == WeightInitialization.NormalizedGaussianRandom)
            {
                double mean = 0;
                double stdDev = 1;
                for (int i = 0; i < W.GetLength(0); i++)
                {
                    for (int j = 0; j < W.GetLength(1); j++)
                    {
                        //W[i, j] = rnd.NextDouble() * 2 - 1; //Old initialization of weights
                        double u1 = 1.0 - rnd.NextDouble(); //uniform(0,1] random doubles
                        double u2 = 1.0 - rnd.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                        W[i, j] = mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
                    }
                }
            }
            else if (_WeightInitialization == WeightInitialization.Random)
            {
                for (int i = 0; i < W.GetLength(0); i++)
                {
                    for (int j = 0; j < W.GetLength(1); j++)
                    {
                        W[i, j] = rnd.NextDouble() * 2 - 1;           
                    }
                }
            }
            //safe previous gradient for adaptive learnrate
            dW_t_minus_1 = new double[W.GetLength(0), W.GetLength(1)];
            //initialize dW_t_minus_1 as zeros-Matrix
            for (int i = 0; i < dW_t_minus_1.GetLength(0); i++)
            {
                for (int j = 0; j < dW_t_minus_1.GetLength(1); j++)
                {
                    dW_t_minus_1[i, j] = 0;
                }
            }
            //initialaze the AdaptiveLearnRateGain
            AdaptiveLearnRateGain = new double[W.GetLength(0), W.GetLength(1)];
            //initialize AdaptiveLearnRateGain as ones-Matrix
            for (int i = 0; i < AdaptiveLearnRateGain.GetLength(0); i++)
            {
                for (int j = 0; j < AdaptiveLearnRateGain.GetLength(1); j++)
                {
                    AdaptiveLearnRateGain[i, j] = 1;
                }
            }
            if (TrainAlgorithm == TrainAlgorithm.Adam)
            {
                //initialize Velocity_Adam as zeros-Matrix
                Velocity_Adam = new double[W.GetLength(0), W.GetLength(1)];
                //initialize Momentum_Adam as zeros-Matrix
                Momentum_Adam = new double[W.GetLength(0), W.GetLength(1)];
            }
            //Z is the matrix that holds output values
            Z = new double[MiniBatchSize, Height * Width * Depth + Bias];//[Row = MiniBatchSize, Column=NeuronsperChannel*NumberOfChannels+Bias]
            //Setting Bias of Z to 1
            for (int i = 0; i < Z.GetLength(0); i++)//Z.GetLenght(0) = BatchSize
            {
                Z[i, Z.GetLength(1) - Bias] = 1;
            }

            Z_3D = new double[Height, Width, Depth* MiniBatchSize];
            //D is the matrix that holds the deltas for this layer
            D = new double[Height * Width * MiniBatchSize, Depth];
            D_3D = new double[Height, Width, Depth * MiniBatchSize];
            //dF is the matrix that holds the derivatives of the activation function
            dF = new double[Height * Width * Depth, MiniBatchSize];
        }
        public void ForewardPropagate(double[,] input)
        {
            if (GetType() == typeof(ConvolutionalLayer))
            {
                PreInput = input;//input = ANNMath.ANNMath.Im2Mat(this[i - 1].Z_3D, this[i].FilterSize, this[i].Stride, true)=>Zeropadded, Biasadded and aligned for upcomming matrixmultiplication
                                 //Compute S of this Layer
                this.S = ANNMath.ANNMath.MultiplyMatrices(input, this.W); //S[Height*Width*MiniBatchSize,Depth]
                //Compute Z of this Layer                                                                      
                for (int b = 0; b < MiniBatchSize; b++)
                {
                    if (ActivationFunction == ActivationFunction.SoftMax)
                    {
                        throw new Exception("SoftMax for ConvLayers not testet. this function should only be used in Outputlayers");
                        //SoftMax_Sum_Exp = 0;
                        //for (int i = 0; i < Depth; i++)
                        //{
                        //    for (int j = 0; j < Height * Width; j++)
                        //    {
                        //        SoftMax_Sum_Exp += Math.Exp(this.S[b * (Height * Width) + j, i]);//Calulating Sum[e^S] for each Example in Minibatch
                        //    }
                        //}
                    }
                    for (int i = 0; i < Depth; i++)
                    {
                        for (int j = 0; j < Height * Width; j++)
                        {
                            this.Z[b, i * (Height * Width) + j] = Activation(this.S[b * (Height * Width) + j, i]);
                        }
                    }
                }
                Z_3D = ANNMath.ANNMath.ComputeZ_3D(this.Z, Z_3D, MiniBatchSize);
                //Compute dF of this Layer
                switch (ActivationFunction)
                {
                    case ActivationFunction.Sigmoid:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < dF.GetLength(0); i++)//dF[Height * Width * Depth, MiniBatchSize] and Z[MiniBatchSize, Height * Width * Depth + Bias]; without Bias
                            {
                                for (int j = 0; j < dF.GetLength(1); j++)
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);// Transpose
                                }
                            }
                            break;
                        }
                    case ActivationFunction.SoftMax:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < dF.GetLength(0); i++)//dF[Height * Width * Depth, MiniBatchSize] and Z[MiniBatchSize, Height * Width * Depth + Bias]; without Bias
                            {
                                for (int j = 0; j < dF.GetLength(1); j++)
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);// Transpose
                                }
                            }
                            break;
                        }
                    case ActivationFunction.Tanh:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < dF.GetLength(0); i++)//dF[Height * Width * Depth, MiniBatchSize] and Z[MiniBatchSize, Height * Width * Depth + Bias]; without Bias
                            {
                                for (int j = 0; j < dF.GetLength(1); j++)
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);//Transpose
                                }
                            }
                            break;
                        }
                    default:
                        {
                            for (int b = 0; b < MiniBatchSize; b++)
                            {
                                for (int i = 0; i < Depth; i++)//dF[Height * Width * Depth, MiniBatchSize] and S[Height*Width*MiniBatchSize,Depth]; without Bias
                                {
                                    for (int j = 0; j < Height * Width; j++)
                                    {
                                        this.dF[i * (Height * Width) + j, b] = Derivative(this.S[b * (Height * Width) + j, i]);//Transpose
                                    }
                                }
                            }
                            break;
                        }
                }
            }
            else//if Hidden- or OutputLayer
            {
                PreInput = input;//Simply input = Z[this-1]
                //Compute S of this Layer
                this.S = ANNMath.ANNMath.MultiplyMatrices(PreInput, this.W);
                //Compute Z of this Layer          
                for (int i = 0; i < this.S.GetLength(0); i++)//S.GetLength(0)=BatchSize
                {                   
                    if (ActivationFunction == ActivationFunction.SoftMax)
                    {
                        SoftMax_Sum_Exp = 0;
                        for (int j = 0; j < this.S.GetLength(1); j++)//S.GetLength(1) =ThisLayersNeurons without Bias
                        {
                            SoftMax_Sum_Exp += Math.Exp(this.S[i, j]);//Calulating Sum[e^S] for each Example in BatchSize
                        }
                    }
                    for (int j = 0; j < this.S.GetLength(1); j++)//S.GetLength(1) =ThisLayersNeurons without Bias
                    {
                        this.Z[i, j] = Activation(this.S[i, j]);
                    }
                }
                Z_3D = null;
                //Compute dF of this Layer
                switch (ActivationFunction)
                {
                    case ActivationFunction.Sigmoid:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < S.GetLength(1); i++)//ThisLayersNeurons=S.GetLength(1)without Bias-->theres no dF for Bias
                            {
                                for (int j = 0; j < S.GetLength(0); j++)//S.GetLength(0)=BatchSize
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);//Transpose
                                }
                            }
                            break;
                        }
                    case ActivationFunction.SoftMax:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < S.GetLength(1); i++)//ThisLayersNeurons=S.GetLength(1)without Bias-->theres no dF for Bias
                            {
                                for (int j = 0; j < S.GetLength(0); j++)//S.GetLength(0)=BatchSize
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);//Transpose
                                }
                            }
                            break;
                        }
                    case ActivationFunction.Tanh:
                        {
                            //the output (Z) of this layer as Input to the derivative function
                            for (int i = 0; i < S.GetLength(1); i++)//ThisLayersNeurons=S.GetLength(1)without Bias-->theres no dF for Bias
                            {
                                for (int j = 0; j < S.GetLength(0); j++)//S.GetLength(0)=BatchSize
                                {
                                    this.dF[i, j] = Derivative(this.Z[j, i]);//Transpose
                                }
                            }
                            break;
                        }
                    default:
                        {
                            for (int i = 0; i < S.GetLength(1); i++)//ThisLayersNeurons=S.GetLength(1)without Bias-->theres no dF for Bias
                            {
                                for (int j = 0; j < S.GetLength(0); j++)//S.GetLength(0)=BatchSize
                                {
                                    this.dF[i, j] = Derivative(this.S[j, i]);//Transpose
                                }
                            }
                            break;
                        }
                }
            }
        }
        public virtual void BackPropagate(double[,] error)
        {

        }
        public virtual void BackPropagate(double[,] OutgoingWeights, double[,] NextLayersD)
        {

        }
        public virtual void BackPropagate(double[,] OutgoingWeightsRotated, int FilterSize, double[,] NextLayersD)
        {

        }
        public void UpdateWeights()
        {
            if (GetType() == typeof(ConvolutionalLayer))
            {
                double[,] dW = ANNMath.ANNMath.MultiplyMatrices(ANNMath.ANNMath.TransposeMatrix(PreInput), D);//new double[W.GetLength(0), W.GetLength(1)];
                //calculating beta for conjugate gradient descent
                if (TrainAlgorithm == TrainAlgorithm.ConjugateGradient)
                {
                    //beta = <(dW-dW_t_minus_1),dW>/(<dW_t_minus_1,dW_t_minus_1> + kleiner Term (10^-7) um divisionen durch 0 zu verhindern);//Polak-Ribiere
                    double beta_numerator = 0;
                    double temp_denumerator = 0;
                    if (beta_denominator == 0)
                    {
                        beta_denominator += 0.0000001;//add 0.0000001 to denominator->never devide by 0!!
                    }
                    for (int o = 0; o < this.W.GetLength(0); o++)
                    {
                        for (int p = 0; p < this.W.GetLength(1); p++)
                        {
                            beta_numerator += (dW[o, p] - dW_t_minus_1[o, p]) * dW[o, p];//Polak-Ribiere
                            temp_denumerator += dW[o, p] * dW[o, p];
                        }
                    }
                    beta = (beta_numerator/(MiniBatchSize*MiniBatchSize)) / beta_denominator;//average over MiniBatch
                    //calculating beta_denominator for next weightupdate time step: dW[o, p] * dW[o, p] in next timestep is dW_t_minus_1[o, p] * dW_t_minus_1[o, p]
                    beta_denominator = temp_denumerator / (MiniBatchSize * MiniBatchSize);//average over MiniBatch
                }
                //calculating new weights
                if (TrainAlgorithm == TrainAlgorithm.Adam)
                {
                    for (int i = 0; i < this.Momentum_Adam.GetLength(0); i++)
                    {
                        for (int j = 0; j < this.Momentum_Adam.GetLength(1); j++)
                        {
                            dW[i, j] = dW[i, j] / MiniBatchSize;//average over MiniBatch
                            if (!first_update)
                            {
                                if (dW[i, j] * dW_t_minus_1[i, j] > 0)//if sign of the gradient dW did not change from dW_t_minus_1 to dW
                                {
                                    if (AdaptiveLearnRateGain[i, j] < 100)//AdaptiveLearnRateGain should not become to high
                                    {
                                        AdaptiveLearnRateGain[i, j] += 0.05;
                                    }
                                }
                                else
                                {
                                    if (AdaptiveLearnRateGain[i, j] > 0.01)//AdaptiveLearnRateGain should not become to low
                                    {
                                        AdaptiveLearnRateGain[i, j] = AdaptiveLearnRateGain[i, j] * 0.95;
                                    }
                                }
                            }
                            Momentum_Adam[i, j] = Momentum_Adam[i, j] * beta1_Adam + (1 - beta1_Adam) * dW[i, j];
                            Velocity_Adam[i, j] = Velocity_Adam[i, j] * beta2_Adam + (1 - beta2_Adam) * dW[i, j] * dW[i, j];

                            double Momentum = Momentum_Adam[i, j] / (1 - Math.Pow(beta1_Adam, timestep_Adam));
                            double Velocity = Velocity_Adam[i, j] / (1 - Math.Pow(beta2_Adam, timestep_Adam));
                            double Adam = Momentum / (Math.Sqrt(Velocity) + epsilon_Adam);
                            if (i < W.GetLength(0) - 1)//no L2-regularization for bias
                            {
                                W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * Adam  + AdaptiveLearnRateGain[i, j] * LearnRate / MiniBatchSize * L2_Regularization * W[i, j];
                            }
                            else//no L2-regularization for bias
                            {
                                W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * Adam ;
                            }                           
                        }
                    }
                    timestep_Adam++;
                }
                else//if not Adam
                {
                    for (int i = 0; i < this.W.GetLength(0); i++)
                    {
                        for (int j = 0; j < this.W.GetLength(1); j++)
                        {
                            dW[i, j] = dW[i, j] / MiniBatchSize;//average over MiniBatch
                            if (!first_update)
                            {
                                if (dW[i, j] * dW_t_minus_1[i, j] > 0)//if sign of the gradient dW did not change from dW_t_minus_1 to dW
                                {
                                    if (AdaptiveLearnRateGain[i, j] < 100)//AdaptiveLearnRateGain should not become to high
                                    {
                                        AdaptiveLearnRateGain[i, j] += 0.05;
                                    }
                                }
                                else
                                {
                                    if (AdaptiveLearnRateGain[i, j] > 0.01)//AdaptiveLearnRateGain should not become to low
                                    {
                                        AdaptiveLearnRateGain[i, j] = AdaptiveLearnRateGain[i, j] * 0.95;
                                    }
                                }
                            }
                            if (TrainAlgorithm == TrainAlgorithm.ConjugateGradient)
                            {
                                dW[i, j] = dW[i, j] + beta * dW_t_minus_1[i, j];//if its the first step, dW_t_minus_1[i, j] = 0 because its initialazed like that                         
                            }
                            if (i < W.GetLength(0) - 1)//no L2-regularization for bias
                            {
                                W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * dW[i, j] + AdaptiveLearnRateGain[i, j] * LearnRate / MiniBatchSize * L2_Regularization * W[i, j];
                            }
                            else//no L2-regularization for bias
                            {
                                W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * dW[i, j];
                            }
                        }
                    }
                }
                if (first_update == true)
                {
                    first_update = false;
                }
                dW_t_minus_1 = dW;
            }
            else//if not convolutional layer
            {
                if (this.GetType() != typeof(InputLayer))
                {
                    double[,] dW = ANNMath.ANNMath.TransposeMatrix(ANNMath.ANNMath.MultiplyMatrices(D, PreInput));
                    //calculating beta for conjugate gradient descent
                    if (TrainAlgorithm == TrainAlgorithm.ConjugateGradient)
                    {
                        //beta = <(dW-dW_t_minus_1),dW>/(<dW_t_minus_1,dW_t_minus_1> + kleiner Term (10^-7) um divisionen durch 0 zu verhindern);//Polak-Ribiere
                        double beta_numerator = 0;
                        double temp_denumerator = 0;
                        if (beta_denominator == 0)
                        {
                            beta_denominator += 0.0000001;//add 0.0000001 to denominator->never devide by 0!!
                        }
                        for (int o = 0; o < this.W.GetLength(0); o++)
                        {
                            for (int p = 0; p < this.W.GetLength(1); p++)
                            {
                                beta_numerator += (dW[o, p] - dW_t_minus_1[o, p]) * dW[o, p];//Polak-Ribiere
                                temp_denumerator += dW[o, p] * dW[o, p];
                            }
                        }
                        beta = (beta_numerator / (MiniBatchSize * MiniBatchSize)) / beta_denominator;//average over MiniBatch
                        //calculating beta_denominator for next weightupdate time step: dW[o, p] * dW[o, p] in next timestep is dW_t_minus_1[o, p] * dW_t_minus_1[o, p]
                        beta_denominator = temp_denumerator / (MiniBatchSize * MiniBatchSize);//average over MiniBatch
                    }
                    //calculating new weights
                    if (TrainAlgorithm == TrainAlgorithm.Adam)
                    {
                        for (int i = 0; i < this.Momentum_Adam.GetLength(0); i++)
                        {
                            for (int j = 0; j < this.Momentum_Adam.GetLength(1); j++)
                            {
                                dW[i, j] = dW[i, j] / MiniBatchSize;//average over MiniBatch
                                if (!first_update)
                                {
                                    if (dW[i, j] * dW_t_minus_1[i, j] > 0)//if sign of the gradient dW did not change from dW_t_minus_1 to dW
                                    {
                                        if (AdaptiveLearnRateGain[i, j] < 100)//AdaptiveLearnRateGain should not become to high
                                        {
                                            AdaptiveLearnRateGain[i, j] += 0.05;
                                        }
                                    }
                                    else
                                    {
                                        if (AdaptiveLearnRateGain[i, j] > 0.01)//AdaptiveLearnRateGain should not become to low
                                        {
                                            AdaptiveLearnRateGain[i, j] = AdaptiveLearnRateGain[i, j] * 0.95;
                                        }
                                    }
                                }
                                Momentum_Adam[i, j] = Momentum_Adam[i, j] * beta1_Adam + (1 - beta1_Adam) * dW[i, j];
                                Velocity_Adam[i, j] = Velocity_Adam[i, j] * beta2_Adam + (1 - beta2_Adam) * dW[i, j] * dW[i, j];

                                double Momentum = Momentum_Adam[i, j] / (1 - Math.Pow(beta1_Adam, timestep_Adam));
                                double Velocity = Velocity_Adam[i, j] / (1 - Math.Pow(beta2_Adam, timestep_Adam));
                                double Adam = Momentum / (Math.Sqrt(Velocity) + epsilon_Adam);
                                if (i < W.GetLength(0) - 1)//no L2-regularization for bias
                                {
                                    W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * Adam + AdaptiveLearnRateGain[i, j] * LearnRate / MiniBatchSize * L2_Regularization * W[i, j];
                                }
                                else//no L2-regularization for bias
                                {
                                    W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * Adam;
                                }                           
                            }
                        }
                        timestep_Adam++;
                    }
                    else//if not Adam
                    {
                        for (int i = 0; i < this.W.GetLength(0); i++)
                        {
                            for (int j = 0; j < this.W.GetLength(1); j++)
                            {
                                dW[i, j] = dW[i, j] / MiniBatchSize;//average over MiniBatch
                                if (!first_update)
                                {
                                    if (dW[i, j] * dW_t_minus_1[i, j] > 0)//if sign of the gradient dW did not change from dW_t_minus_1 to dW
                                    {
                                        if (AdaptiveLearnRateGain[i, j] < 100)//AdaptiveLearnRateGain should not become to high
                                        {
                                            AdaptiveLearnRateGain[i, j] += 0.05;
                                        }
                                    }
                                    else
                                    {
                                        if (AdaptiveLearnRateGain[i, j] > 0.01)//AdaptiveLearnRateGain should not become to low
                                        {
                                            AdaptiveLearnRateGain[i, j] = AdaptiveLearnRateGain[i, j] * 0.95;
                                        }
                                    }
                                }
                                if (TrainAlgorithm == TrainAlgorithm.ConjugateGradient)
                                {
                                    dW[i, j] = dW[i, j] + beta * dW_t_minus_1[i, j];//if its the first step, dW_t_minus_1[i, j] = 0 because its initialazed like that
                                }
                                if (i < W.GetLength(0) - 1)//no L2-regularization for bias
                                {
                                    W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * dW[i, j] + AdaptiveLearnRateGain[i, j] * LearnRate / MiniBatchSize * L2_Regularization * W[i, j];
                                }
                                else//no L2-regularization for bias
                                {
                                    W[i, j] -= LearnRate / MiniBatchSize * AdaptiveLearnRateGain[i, j] * dW[i, j];
                                }
                            }
                        }                                
                    }
                    if (first_update == true)
                    {
                        first_update = false;
                    }
                    dW_t_minus_1 = dW;
                }
            }
        }
 
        private double Activation(double Input)
        {
            if (ActivationFunction == ActivationFunction.Sigmoid)
            {
                return 1 / (1 + Math.Exp(-gradient * (Input)));   //trifft für alle Neuronen ausser den Inputneuronen zu, da nur bei den Inputneuronen die Variable Neuron.output von double.MinValue abgeändert wurden
                                                                                        //bei allen anderen Neuronen hat diese Variable (Neuron.output) immer den Wert double.MinValue
                                                                                        //bias geht auf jedes neuron, was wichtig ist, um inputs die null sind verwerten zu können)
            }
            else if (ActivationFunction == ActivationFunction.ReLu)
            {
                if (Input >= 0)
                {
                    return Input;
                }
                else
                {
                    return 0;
                }
            }
            else if (ActivationFunction == ActivationFunction.lReLu)
            {
                if (Input >= 0)
                {
                    return Input;
                }
                else
                {
                    return lReluByPassFactor* Input;
                }
            }
            else if (ActivationFunction == ActivationFunction.SoftPlus)
            {
                return Math.Log(1 + Math.Exp(gradient * (Input)));
            }
            else if (ActivationFunction == ActivationFunction.Tanh)
            {
                return (1 - Math.Exp(-gradient * 2 * (Input))) / (1 + Math.Exp(-gradient * 2 * (Input)));
            }
            else if (ActivationFunction == ActivationFunction.SoftMax)
            {
                return Math.Exp(Input) / SoftMax_Sum_Exp;
            }
            else
            {
                return 1 / (1 + Math.Exp(-gradient * (Input)));//Default is Sigmoid
            }
        }
        private double Derivative(double Input)
        {
            if (ActivationFunction == ActivationFunction.Sigmoid)
            {
                return Input * (1 - Input);
            }
            else if (ActivationFunction == ActivationFunction.ReLu)
            {
                if (Input > 0)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
            else if (ActivationFunction == ActivationFunction.lReLu)
            {
                if (Input > 0)
                {
                    return 1;
                }
                else
                {
                    return lReluByPassFactor;
                }
            }
            else if (ActivationFunction == ActivationFunction.SoftPlus)
            {
                return 1 / (1 + Math.Exp(-gradient * (Input)));
            }
            else if (ActivationFunction == ActivationFunction.Tanh)
            {
                return 1 - Input * Input;
            }
            else if (ActivationFunction == ActivationFunction.SoftMax)
            {
                return Input * (1 - Input);
            }
            else
            {
                return Input * (1 - Input);
            }
        }    
    }
  
    public enum ActivationFunction
    {
        None,
        Sigmoid,
        Tanh,
        lReLu,
        ReLu,
        SoftMax,
        SoftPlus
    }
    public enum WeightInitialization
    {
        Random, NormalizedGaussianRandom
    }
}
