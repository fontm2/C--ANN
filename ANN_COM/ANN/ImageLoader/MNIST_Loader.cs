using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;


namespace ImageLoader
{
    public class MNIST_Loader
    {
        private BinaryReader br;
        private byte[] b_read_images;
        private byte[] b_read_labels;
        int labels_read_offset = 8;
        int images_read_offset = 16;
        private int MiniBatchSize;
        double[,] labels;
        double[,,] z_3D;

        public MNIST_Loader(string[] FileNames, int _MiniBatchSize, bool train)
        {
            MiniBatchSize = _MiniBatchSize;
            if (train)
            {               
                //get trainingData
                br = new BinaryReader(File.Open(FileNames[0], FileMode.Open));
                b_read_images = new byte[47040016];
                int readBytes_Images = br.Read(b_read_images, 0, 47040016);
                br.Close();
                br = new BinaryReader(File.Open(FileNames[1], FileMode.Open));
                b_read_labels = new byte[60008];
                int readBytes_Labels = br.Read(b_read_labels, 0, 60008);
                br.Close();
            }
            else
            {
                //get testData
                br = new BinaryReader(File.Open(FileNames[2], FileMode.Open));
                b_read_images = new byte[7840016];
                int readBytes_Images = br.Read(b_read_images, 0, 7840016);
                br.Close();
                br = new BinaryReader(File.Open(FileNames[3], FileMode.Open));
                b_read_labels = new byte[10008];
                int readBytes_Labels = br.Read(b_read_labels, 0, 10008);
                br.Close();
            }
        }
        public double[,] GetLabels(int BatchNum)
        {
            #region sort Labels
            labels = new double[MiniBatchSize, 10];//filled with zeros
            for (int j = 0; j < labels.GetLength(0); j++)
            {
                int LabelNr = Convert.ToInt32(b_read_labels[labels_read_offset + BatchNum * MiniBatchSize + j]);
                switch (LabelNr)
                {
                    case 0:
                        {
                            labels[j, 0] = 1;
                            break;
                        }
                    case 1:
                        {
                            labels[j, 1] = 1;
                            break;
                        }
                    case 2:
                        {
                            labels[j, 2] = 1;
                            break;
                        }
                    case 3:
                        {
                            labels[j, 3] = 1;
                            break;
                        }
                    case 4:
                        {
                            labels[j, 4] = 1;
                            break;
                        }
                    case 5:
                        {
                            labels[j, 5] = 1;
                            break;
                        }
                    case 6:
                        {
                            labels[j, 6] = 1;
                            break;
                        }
                    case 7:
                        {
                            labels[j, 7] = 1;
                            break;
                        }
                    case 8:
                        {
                            labels[j, 8] = 1;
                            break;
                        }
                    case 9:
                        {
                            labels[j, 9] = 1;
                            break;
                        }
                }
            }
            #endregion
            return labels;
        }
        public double[,,] GetZ_3D(int BatchNum)
        {
            #region Creat InputLayer.Z_3D[height, width, depth * MiniBatchSize] matrix
            //Per Example in MiniBatch, the Depth-Order ist Blue, Green, Red

            byte[] temp = new byte[784];//28*28=784
            z_3D = new double[27, 27, 1 * MiniBatchSize];
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        temp[i * 28 + j] = b_read_images[images_read_offset + BatchNum * MiniBatchSize * 28 * 28 + i * 28 + j + b * 28 * 28];
                    }
                }
                for (int i = 0; i < 27; i++)//only 27
                {
                    for (int j = 0; j < 27; j++)//only 27
                    {
                        z_3D[i, j, 1 * b] = Convert.ToDouble(temp[(i + 1) * 28 + j]);
                    }
                }
            }

            #endregion
            return z_3D;
        }
    }
}
