using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageLoader
{
    public class BinLoader
    {
        private BinaryReader br;
        int MiniBatchSize = 0;
        int Height = 31;
        int Witth = 31;
        int Depth = 3;
        int LabelOffset = 1;
        byte[] b_read;
        byte[] b_red = new byte[1024];
        byte[] b_green = new byte[1024];
        byte[] b_blue = new byte[1024];
        int bytesPerPicturInclLabel = 3073;
        double[,] labels;
        double[,,] z_3D;

        public BinLoader(string FileName, int _MiniBatchSize)
        {
            MiniBatchSize = _MiniBatchSize;
            br = new BinaryReader(File.Open(FileName, FileMode.Open));
            b_read = new byte[30730000];
            int readBytes = br.Read(b_read, 0, 30730000);
            br.Close();
        }

        public double[,] GetLabels(int BatchNum)
        {
            #region sort Labels
            labels = new double[MiniBatchSize, 10];//filled with zeros
            for (int j = 0; j < labels.GetLength(0); j++)
            {
                int LabelNr = Convert.ToInt32(b_read[BatchNum * MiniBatchSize * bytesPerPicturInclLabel + j * bytesPerPicturInclLabel]);//BatchNum * MiniBatchSize * bytesPerPicturInclLabel invrements the amount of bytes for current Batchnumber
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
            z_3D = new double[Height, Witth, Depth * MiniBatchSize];

            for (int b = 0; b < MiniBatchSize; b++)
            {
                //blue
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        b_blue[i * 32 + j] = b_read[BatchNum * MiniBatchSize * bytesPerPicturInclLabel + LabelOffset + 2 * 32 * 32 + i * 32 + j + b * bytesPerPicturInclLabel];//BatchNum * MiniBatchSize * bytesPerPicturInclLabel invrements the amount of bytes for current Batchnumber
                    }
                }
                for (int i = 0; i < 31; i++)
                {
                    for (int j = 0; j < 31; j++)
                    {
                        z_3D[i, j, b * Depth] = Convert.ToDouble(b_blue[i * 32 + j]);//not yet normalized
                    }
                }
                //green
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        b_green[i * 32 + j] = b_read[BatchNum * MiniBatchSize * bytesPerPicturInclLabel + LabelOffset + 1 * 32 * 32 + i * 32 + j + b * bytesPerPicturInclLabel];//BatchNum * MiniBatchSize * bytesPerPicturInclLabel invrements the amount of bytes for current Batchnumber
                    }
                }
                for (int i = 0; i < 31; i++)
                {
                    for (int j = 0; j < 31; j++)
                    {
                        z_3D[i, j, b * Depth + 1] = Convert.ToDouble(b_green[i * 32 + j]);//not yet normalized
                    }
                }

                //red
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        b_red[i * 32 + j] = b_read[BatchNum * MiniBatchSize * bytesPerPicturInclLabel + LabelOffset + i * 32 + j + b * bytesPerPicturInclLabel];//BatchNum * MiniBatchSize * bytesPerPicturInclLabel invrements the amount of bytes for current Batchnumber
                    }
                }
                for (int i = 0; i < 31; i++)
                {
                    for (int j = 0; j < 31; j++)
                    {
                        z_3D[i, j, b * Depth + 2] = Convert.ToDouble(b_red[i * 32 + j]);//not yet normalized
                    }
                }
            }
            #endregion
            return z_3D;
        }
    }
}
