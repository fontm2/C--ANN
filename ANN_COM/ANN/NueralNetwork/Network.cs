using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANNMath;
using ImageLoader;

namespace NueralNetwork
{
    public class Network: List<Layer>
    {
        public delegate void SaveWeights(Network Net);
        public event SaveWeights SaveWeightsEvent;
        public delegate void BadgeError(Network Net, double error);
        public event BadgeError BadgeErrorEvent;
        public delegate void NetAccuracy(Network Net, double accuracy);
        public event NetAccuracy NetAccuracyEvent;
        private double[,] Labels;
        private int counter = 0;
        public CostFunction CostFunction;
        private ImageLoader.BinLoader Bl = null;
        private int epoche = 0;
        private int MiniBatchSize;
        string[] ListOfFilenames;
        private int EpocheMiniBatchCounter = 1;
        private int BatchCounter = 0;
        double accumulatedError = 0;
        private int NoOfImages;
        private MNIST_Loader MNIST_Load;
        private bool IsMNIST;
        private int how_Mane_Epoches_per_Minibatch = 1;//if 1, it changes minibatch on each epoche
        private double L2_reg;
        private double L2_reg_Temp;

        public Network()//MLP
        {
            int MiniBatchSize = 40;
            CostFunction = CostFunction.Quadratic;
            Add(new InputLayer(new double[MiniBatchSize, 784], MiniBatchSize));          
            Add(new HiddenLayer(this[Count - 1].Z, 1000, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.04, 0.000025, 0.1));
            Add(new HiddenLayer(this[Count - 1].Z, 1000, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.04, 0.000025, 0.1));
            Add(new OutputLayer(this[Count - 1].Z, 10, MiniBatchSize, ActivationFunction.SoftPlus, WeightInitialization.Random, TrainAlgorithm.SGD, 0.04, 0.000025, 0.1));
            Labels = new double[MiniBatchSize, 10];          
        }
        public Network(string[] InputFileNames, double[,] Outputs)//ConvNet with 4 channel png-images
        {
            CostFunction = CostFunction.CrossEntropy;
            //Outputs[MiniBatchSize, NumberOfOutputNeurons]
            if (InputFileNames.Length == Outputs.GetLength(0))
            {
                int MiniBatchSize = InputFileNames.Length;
                Labels = Outputs;
                Add(new InputLayer(ImageLoader.ImageHandler.ImageToZ_3D(InputFileNames), MiniBatchSize));
                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 5, 2, 10, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));
                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 3, 1, 10, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.001 * MiniBatchSize,  0.01 / MiniBatchSize, 1));              
                Add(new HiddenLayer(this[Count - 1].Z, 100, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.001 * MiniBatchSize, 0.1 / MiniBatchSize, 1));
                Add(new OutputLayer(this[Count - 1].Z, Labels.GetLength(1), MiniBatchSize, ActivationFunction.SoftMax, WeightInitialization.Random, TrainAlgorithm.SGD, 0.001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));
            }
        }
        public Network(string InputFileName, int _MiniBatchSize, string[] _ListOfFilenames, bool MNIST)//ConvNet with binarized, labelled image-dataset
        {
            L2_reg = 0.01;
            IsMNIST = MNIST;
            if (!MNIST)
            {
                NoOfImages = 10000;//"images per fileName-location" (data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5,test_batch contain each 10000 images)
                MiniBatchSize = _MiniBatchSize;
                ListOfFilenames = _ListOfFilenames;
                CostFunction = CostFunction.CrossEntropy;
                Bl = new ImageLoader.BinLoader(InputFileName, MiniBatchSize);
                Labels = Bl.GetLabels(0);
                Add(new InputLayer(Bl.GetZ_3D(0), MiniBatchSize));
                //long trained network
                //Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 5, 2, 10, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));
                //Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 5, 3, 16, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));
                //Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 3, 1, 16, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));
                //Add(new HiddenLayer(this[Count - 1].Z, 120, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.1 / MiniBatchSize, 1));
                //Add(new HiddenLayer(this[Count - 1].Z, 84, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.1 / MiniBatchSize, 1));
                //Add(new OutputLayer(this[Count - 1].Z, Labels.GetLength(1), MiniBatchSize, ActivationFunction.SoftMax, WeightInitialization.Random, TrainAlgorithm.Adam, 0.00001 * MiniBatchSize, 0.01 / MiniBatchSize, 1));

                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 3, 1, 10, MiniBatchSize, ActivationFunction.lReLu, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.00001 * MiniBatchSize, L2_reg, 1));              
                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 5, 2, 24, MiniBatchSize, ActivationFunction.lReLu, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.00001 * MiniBatchSize, L2_reg, 1));
                Add(new HiddenLayer(this[Count - 1].Z, 120, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.00001 * MiniBatchSize, L2_reg, 1));
                Add(new HiddenLayer(this[Count - 1].Z, 84, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.00001 * MiniBatchSize, L2_reg, 1));
                Add(new OutputLayer(this[Count - 1].Z, Labels.GetLength(1), MiniBatchSize, ActivationFunction.SoftMax, WeightInitialization.Random, TrainAlgorithm.SGD, 0.00001 * MiniBatchSize, L2_reg, 1));
            }
            else//if MNIST-Data
            {
                ListOfFilenames = _ListOfFilenames;
                NoOfImages = 60000;//"images per fileName-location"(t10k-images.idx3-ubyte contains 60000 images)
                MiniBatchSize = _MiniBatchSize;
                CostFunction = CostFunction.CrossEntropy;
                MNIST_Load = new ImageLoader.MNIST_Loader(ListOfFilenames, MiniBatchSize, true);
                Labels = MNIST_Load.GetLabels(0);
                Add(new InputLayer(MNIST_Load.GetZ_3D(0), MiniBatchSize));
                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 3, 1, 6, MiniBatchSize, ActivationFunction.lReLu, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.0001 * MiniBatchSize, L2_reg, 1));
                Add(new ConvolutionalLayer(this[Count - 1].Z_3D, 3, 2, 16, MiniBatchSize, ActivationFunction.lReLu, WeightInitialization.NormalizedGaussianRandom, TrainAlgorithm.SGD, 0.0001 * MiniBatchSize, L2_reg, 1));                
                Add(new HiddenLayer(this[Count - 1].Z, 120, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.0001 * MiniBatchSize, L2_reg, 1));
                Add(new HiddenLayer(this[Count - 1].Z, 84, MiniBatchSize, ActivationFunction.Sigmoid, WeightInitialization.Random, TrainAlgorithm.SGD, 0.0001 * MiniBatchSize, L2_reg, 1));
                Add(new OutputLayer(this[Count - 1].Z, Labels.GetLength(1), MiniBatchSize, ActivationFunction.SoftMax, WeightInitialization.Random, TrainAlgorithm.SGD, 0.0001 * MiniBatchSize, L2_reg, 1));
            }           
        }


        public void Load(string file)//Load Datat for MLP
        {
            StreamReader reader = File.OpenText(file);
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();           
                string[] lines = line.Split(',');
                if (counter < 40)
                {
                    for (int i = 0; i < 784; i++)
                    {
                        this[0].Z[counter, i] = double.Parse(lines[i]);
                    }
                    for (int i = 0; i < 10; i++)
                    {
                        Labels[counter, i] = double.Parse(lines[i + 784]);
                    }
                }
                counter++;
            }
            reader.Close();
        }
        public void Activate()
        {
            for (int i = 1; i < this.Count; i++)//the InputLayer got it's Output (Z) set, when loading the inputdata aka calling Load Methode
            {
                if (this[i].GetType() == typeof(ConvolutionalLayer) && this[i - 1].Z_3D != null)
                {
                    this[i].ForewardPropagate(ANNMath.ANNMath.Im2Mat(this[i - 1].Z_3D, this[i].FilterSize, this[i].Stride, true, this[i].MiniBatchSize));//Im2Mat fügt Zeropadding und Bias an falls true
                }
                else
                {
                    this[i].ForewardPropagate(this[i - 1].Z);
                }
            }
        }
        public double BackProp()
        {
            double error = 0;
            double[,] errors = new double[this[this.Count - 1].Z.GetLength(0), this[this.Count - 1].Z.GetLength(1)];
            //Get the Error, by subtracting desired Values from Output (Z) of OutputLayer. 
            //The OutputLayer has no Bias in Z
            if (this[this.Count - 1].Z.GetLength(0) == Labels.GetLength(0) && this[this.Count - 1].Z.GetLength(1) == Labels.GetLength(1))
            {
                for (int i = 0; i < this[this.Count - 1].Z.GetLength(0); i++)
                {
                    for (int j = 0; j < this[this.Count - 1].Z.GetLength(1); j++)
                    {
                        switch(CostFunction)
                        {
                            //errors[i,j] contains the term d(error(out))/d(out)
                            case CostFunction.Quadratic:
                                {
                                    errors[i, j] = this[this.Count - 1].Z[i, j] - Labels[i, j];//d(QuadraticError)/d(out)
                                    error += Math.Pow(errors[i, j], 2) / 2;//Quadratic error
                                    break;
                                }
                            case CostFunction.CrossEntropy:
                                {
                                    if (this[this.Count - 1].ActivationFunction == ActivationFunction.Sigmoid)//the use binary Corss Entropy
                                    {
                                        errors[i, j] = (this[this.Count - 1].Z[i, j] - Labels[i, j]) / (this[this.Count - 1].Z[i, j] * (1 - this[this.Count - 1].Z[i, j])); //d(binaryCrossEntropy)/d(out)
                                        error += -(Labels[i, j] * Math.Log(this[this.Count - 1].Z[i, j]) - (1 - Labels[i, j]) * Math.Log(1 - this[this.Count - 1].Z[i, j]));//binary Cross Entropy
                                    }
                                    else if (this[this.Count - 1].ActivationFunction == ActivationFunction.SoftMax)
                                    {
                                        errors[i, j] = -Labels[i, j] / this[this.Count - 1].Z[i, j];//d(gerneralizedCrossentropy)/d(out)
                                        error += -Labels[i, j] * Math.Log(this[this.Count - 1].Z[i, j]); //gerneralized Cross entropy
                                    }
                                    break;
                                }
                            case CostFunction.Generalized_KL_Divergence:
                                {
                                    errors[i, j] = (this[this.Count - 1].Z[i, j] + Labels[i, j]) / this[this.Count - 1].Z[i, j];
                                    error += Labels[i, j] * Math.Log(Labels[i, j]/ this[this.Count - 1].Z[i, j])- Labels[i, j] + this[this.Count - 1].Z[i, j];
                                    break;
                                }
                        }
                    }
                }          
                this[this.Count - 1].BackPropagate(ANNMath.ANNMath.TransposeMatrix(errors));//Caluclate D for OutputLayer
                for (int i = this.Count - 2; i > 0; i--)//-2 because last layer is allready done
                {
                    if (this[i + 1].GetType() == typeof(ConvolutionalLayer))
                    {
                        this[i].BackPropagate(this[i + 1].W, this[i + 1].FilterSize, ANNMath.ANNMath.Im2Mat(this[i + 1].D_3D, this[i + 1].FilterSize, this[i + 1].Stride, false, this[i].MiniBatchSize));//Calclulate D for all the ConvLayers from whoms perspictive the nextlayer is convolutional too                                                                                                                                      
                    }
                    else
                    {
                        this[i].BackPropagate(this[i + 1].W, this[i + 1].D);//Calclulate D for all the HiddenLayers or first (on BackPropView) ConvLayer
                    }
                }
            }
            else
            {
                throw new Exception("xxxx");
            }
            //calcluating sum of all Weights^2 for Lossfunction with L2-Regularization
            double SumWeights_Squared = 0.0;
            for (int i = 1; i < this.Count; i++)//i=1, because input layer contains no weights
            {
                for (int j = 0; j < this[i].W.GetLength(0) - 1; j++)//this[i].W.GetLength(0) - 1 because Bias should not be regularized and Bias sits on the bottom of Dimension 0 of the Weight-Matrix W
                {
                    for (int k = 0; k < this[i].W.GetLength(1); k++)
                    {
                        SumWeights_Squared += this[i].W[j, k] * this[i].W[j, k];
                    }
                }
            }
            //L2_reg_Temp is switched betwen 0 whil testing and L2_reg while training
            return (error + L2_reg_Temp * L2_reg * SumWeights_Squared/2)/MiniBatchSize;//average the error over the minibatches
        }
        public void UpdateWeights()
        {
            for (int i = 1; i < this.Count; i++)//the InputLayer has no ingoing Weights to adjust
            {              
                this[i].UpdateWeights();             
            }
        }
        public double Train()
        {
            if (!IsMNIST)
            {
                //int how_Mane_Epoches_per_Minibatch = 1;//if 1, it changes minibatch on each epoche
                if (epoche % (NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) == 0 && epoche > 0)
                {
                    if (EpocheMiniBatchCounter == 5)//ListOfFilenames[0] to ListOfFilenames[4] are training images, ListOfFilenames[5] are test images
                    {
                        EpocheMiniBatchCounter = 0;
                    }                   
                    //Bl = new ImageLoader.BinLoader(ListOfFilenames[EpocheMiniBatchCounter], MiniBatchSize);//is replaced at the end of the TestNetwork-Method
                    EpocheMiniBatchCounter++;
                    BatchCounter++;
                    OnBadgeErrorEvent(accumulatedError / (NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch));
                    accumulatedError = 0;
                    OnSaveWeightsEvent();
                    TestNetwork();
                }
                if (Bl != null)
                {
                    //change MiniBatch all "how_Mane_Epoches_per_Minibatch" Epoches
                    if (epoche % how_Mane_Epoches_per_Minibatch == 0)
                    {
                        Labels = Bl.GetLabels((epoche - BatchCounter * NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) / how_Mane_Epoches_per_Minibatch);
                        this[0].Z_3D = Bl.GetZ_3D((epoche - BatchCounter * NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) / how_Mane_Epoches_per_Minibatch);
                    }
                }
                //für jeden neuen Durchgang (aufgerufen aus MainWindwo.xmal per TrainNetwork()) wird der fehler auf 0 gesetzt
                Activate();
                double error = BackProp();
                UpdateWeights();
                epoche++;

                //OnBadgeErrorEvent(error);
                //accumulatedError = 0;

                accumulatedError += error;
                return error;
            }
            else//if MNIST-Data
            {
                //int how_Mane_Epoches_per_Minibatch = 1;//if 1, it changes minibatch on each epoche
                if (epoche % (NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) == 0 && epoche > 0)
                {                  
                    BatchCounter++;
                    OnBadgeErrorEvent(accumulatedError / (NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch));
                    accumulatedError = 0;
                    OnSaveWeightsEvent();
                    TestNetwork();
                }
                if (MNIST_Load != null)
                {
                    //change MiniBatch all "how_Mane_Epoches_per_Minibatch" Epoches
                    if (epoche % how_Mane_Epoches_per_Minibatch == 0)
                    {
                        Labels = MNIST_Load.GetLabels((epoche - BatchCounter * NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) / how_Mane_Epoches_per_Minibatch);
                        this[0].Z_3D = MNIST_Load.GetZ_3D((epoche - BatchCounter * NoOfImages / MiniBatchSize * how_Mane_Epoches_per_Minibatch) / how_Mane_Epoches_per_Minibatch);
                    }
                }
                //für jeden neuen Durchgang (aufgerufen aus MainWindwo.xmal per TrainNetwork()) wird der fehler auf 0 gesetzt
                Activate();
                double error = BackProp();
                UpdateWeights();
                epoche++;

                //OnBadgeErrorEvent(error);
                //accumulatedError = 0;

                accumulatedError += error;
                return error;
            }
        }
        public void TestNetwork()
        {
            L2_reg_Temp = 0;
            if (!IsMNIST)
            {
                int correct_Estimations = 0;
                //load test images 
                Bl = new ImageLoader.BinLoader(ListOfFilenames[5], MiniBatchSize);//ListOfFilenames[5] are test images
                for (int b = 0; b < 10000 / MiniBatchSize; b++)//10000 num of Test Pictures
                {
                    Labels = Bl.GetLabels(b);
                    this[0].Z_3D = Bl.GetZ_3D(b);
                    Activate();
                    for (int i = 0; i < this[this.Count - 1].Z.GetLength(0); i++)// this[this.Count - 1] = OutputLayer || Dimension[0] is MiniBatchSite
                    {
                        int IndexOfEstimatedNumber = 0;
                        double maxOutputval = 0;
                        int IndexOfCorrectLabel = 0;
                        int tempLapel = 0;
                        //get index of output-value with highest probability
                        for (int j = 0; j < this[this.Count - 1].Z.GetLength(1); j++)// this[this.Count - 1] = OutputLayer || Dimension[1] are Labels
                        {
                            if (this[this.Count - 1].Z[i, j] > maxOutputval)
                            {
                                maxOutputval = this[this.Count - 1].Z[i, j];
                                IndexOfEstimatedNumber = j;
                            }
                            if (Labels[i, j] > tempLapel)
                            {
                                tempLapel = (int)Labels[i, j];
                                IndexOfCorrectLabel = j;
                            }
                        }
                        if (IndexOfEstimatedNumber == IndexOfCorrectLabel)
                        {
                            correct_Estimations++;
                        }
                    }
                    OnNetAccuracyEvent((double)correct_Estimations / ((double)((b + 1) * MiniBatchSize)));
                }
                Bl = new ImageLoader.BinLoader(ListOfFilenames[EpocheMiniBatchCounter], MiniBatchSize);
            }
            else
            {
                int correct_Estimations = 0;
                //load test images 
                MNIST_Load = new ImageLoader.MNIST_Loader(ListOfFilenames, MiniBatchSize, false);
                for (int b = 0; b < 10000 / MiniBatchSize; b++)//10000 num of Test Pictures
                {
                    Labels = MNIST_Load.GetLabels(b);
                    this[0].Z_3D = MNIST_Load.GetZ_3D(b);
                    Activate();
                    for (int i = 0; i < this[this.Count - 1].Z.GetLength(0); i++)// this[this.Count - 1] = OutputLayer || Dimension[0] is MiniBatchSite
                    {
                        int IndexOfEstimatedNumber = 0;
                        double maxOutputval = 0;
                        int IndexOfCorrectLabel = 0;
                        int tempLapel = 0;
                        //get index of output-value with highest probability
                        for (int j = 0; j < this[this.Count - 1].Z.GetLength(1); j++)// this[this.Count - 1] = OutputLayer || Dimension[1] are Labels
                        {
                            if (this[this.Count - 1].Z[i, j] > maxOutputval)
                            {
                                maxOutputval = this[this.Count - 1].Z[i, j];
                                IndexOfEstimatedNumber = j;
                            }
                            if (Labels[i, j] > tempLapel)
                            {
                                tempLapel = (int)Labels[i, j];
                                IndexOfCorrectLabel = j;
                            }
                        }
                        if (IndexOfEstimatedNumber == IndexOfCorrectLabel)
                        {
                            correct_Estimations++;
                        }
                    }
                    OnNetAccuracyEvent((double)correct_Estimations / ((double)((b + 1) * MiniBatchSize)));
                }
                //load again TrainImages
                MNIST_Load = new ImageLoader.MNIST_Loader(ListOfFilenames, MiniBatchSize, true);
            }
            L2_reg_Temp = L2_reg;
        }
        public void OnSaveWeightsEvent()
        {
            if (SaveWeightsEvent != null)//checks weather the event has subscribers
            {
                SaveWeightsEvent(this);
            }
        }
        public void OnBadgeErrorEvent(double accumulatedError)
        {
            if (BadgeErrorEvent != null)//checks weather the event has subscribers
            {
                BadgeErrorEvent(this, accumulatedError);
            }
        }
        public void OnNetAccuracyEvent(double accuracy)
        {
            if (NetAccuracyEvent != null)//checks weather the event has subscribers
            {
                NetAccuracyEvent(this, accuracy);
            }
        }
    }
    public enum CostFunction
    {
        Quadratic,
        CrossEntropy,
        Generalized_KL_Divergence
    }
    public enum TrainAlgorithm
    {
        SGD,
        ConjugateGradient,
        Adam
    }
}
