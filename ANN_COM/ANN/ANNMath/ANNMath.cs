using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNMath
{
    public static class ANNMath
    {
        private static int Bias = 1;
        public static double[,] MultiplyMatrices(double[,] mata, double[,] matb)
        {
            int aRows = mata.GetLength(0);
            int aCols = mata.GetLength(1);
            int bRows = matb.GetLength(0);
            int bCols = matb.GetLength(1);
            if (aCols != bRows)
                throw new Exception("xxxx");

            double[,] result = new double[aRows, bCols];

            Parallel.For(0, aRows, i =>
            {
                for (int j = 0; j < bCols; ++j) // each col of B
                    for (int k = 0; k < aCols; ++k) // could use k < bRows
                        result[i,j] += mata[i,k] * matb[k,j];
            }
            );
            return result;
        }
        public static double[,] ElementwiseMultiplication(double[,] mata, double[,] matb)
        {
            int aRows = mata.GetLength(0);
            int aCols = mata.GetLength(1);
            int bRows = matb.GetLength(0);
            int bCols = matb.GetLength(1);
            int totalElements = aRows * aCols;
            if (aRows != bRows || aCols != bCols || (aRows != bRows && aCols != bCols))
                throw new Exception("xxxx");

            double[,] result = new double[aRows, aCols];
            if (totalElements < 5000000)//5*10^6
            {
                for (int i = 0; i < aRows; i++)
                {
                    for (int j = 0; j < aCols; j++)
                    {
                        result[i, j] = mata[i, j] * matb[i, j];
                    }
                }
            }
            else
            {
                Parallel.For(0, aRows, i =>
                {
                    for (int j = 0; j < bCols; ++j) // each col of B                
                        result[i, j] = mata[i, j] * matb[i, j];
                });
            }

            return result;
        }
        public static double[,] TransposeMatrix(double[,] mat)
        {
            double[,] Transpose = new double[mat.GetLength(1), mat.GetLength(0)];
            for (int i = 0; i < Transpose.GetLength(0); i++)
            {
                for (int j = 0; j < Transpose.GetLength(1); j++)
                {
                    Transpose[i, j] = mat[j, i];
                }
            }
            return Transpose;
        }
        public static double[,] MultiplyMatrixWithScalar(double scalar, double[,] mat)
        {
            int Rows = mat.GetLength(0);
            int Cols = mat.GetLength(1);
            double[,] result = new double[Rows, Cols];
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    result[i, j] = mat[i, j] * scalar;
                }
            }
            return result;
        }

        public static double[,] Im2Mat(double[,,] Input, int FilterSize, int Stride, bool CallFromForewardPass, int MiniBatchSize)
        {
            #region Foreward
            if (CallFromForewardPass)
            {
                //for Z_3D[i-1] as Input-> Bias (ForewardPass)      
                //Input.GetLength(0) = Height(i-1)
                //Input.GetLength(1) =Width(i-1)
                //Input.GetLength(2) = Depth(i-1)*MiniBacthSize = DepthOfInputLayer*MiniBatchSize
                int Height = (Input.GetLength(0) - 1) / Stride + 1;//when changing something here, also change in ConvLayer's Constructor
                int Width = (Input.GetLength(1) - 1) / Stride + 1;//when changing something here, also change in ConvLayer's Constructor
                #region ZeroPadding
                int ZeroPadding = (FilterSize - 1) / 2;
                double[,,] InputZeroPadded = new double[Input.GetLength(0) + 2 * ZeroPadding, Input.GetLength(1) + 2 * ZeroPadding, Input.GetLength(2)]; //InputZeroPadded is filled with Zeros
                for (int i = 0; i < Input.GetLength(0); i++)
                {
                    for (int j = 0; j < Input.GetLength(1); j++)
                    {
                        for (int k = 0; k < Input.GetLength(2); k++)
                        {
                            InputZeroPadded[i + ZeroPadding, j + ZeroPadding, k] = Input[i, j, k];
                        }
                    }
                }
                #endregion
                //Output wird erst beim return key-word transponiert
                double[,] Output = new double[FilterSize * FilterSize * InputZeroPadded.GetLength(2) / MiniBatchSize + Bias, Height * Width * MiniBatchSize];//[ThisFilterSizeze*ThisFilterSizeze*PreviousLayersDepth, ThisHeight*ThisWidth*MiniBatchSize]
                #region rearranging Input
                //calculating Output
                for (int b = 0; b < MiniBatchSize; b++)
                {
                    for (int j = 0; j < Width; j++)
                    {
                        for (int i = 0; i < Height; i++)
                        {
                            for (int k = 0; k < InputZeroPadded.GetLength(2) / MiniBatchSize; k++)//DepthOfInputLayer=InputZeroPadded.GetLength(2)/MiniBatchSize
                            {
                                for (int jj = 0; jj < FilterSize; jj++)
                                {
                                    for (int ii = 0; ii < FilterSize; ii++)
                                    {
                                        Output[k * (FilterSize * FilterSize) + jj * FilterSize + ii, b * (Height * Width) + j * Height + i] = InputZeroPadded[i * Stride + ii, j * Stride + jj, k + b * InputZeroPadded.GetLength(2) / MiniBatchSize];
                                    }
                                }
                            }
                            //Bias
                            Output[Output.GetLength(0) - Bias, b * (Height * Width) + j * Height + i] = 1;
                        }
                    }
                }
                #endregion
                return TransposeMatrix(Output);//Muss Bias und Zeropadding haben haben, sowie Transponiert werden
            }
            #endregion
            #region Backward
            else //Backpropagation
            {
                //for D_3D[i+1] as Input-> no Bias (BackwarPass)
                //Input.GetLength(0) = Height
                //Input.GetLength(1) =Width
                //Input.GetLength(2) = Depth*MiniBacthSize = DepthOfThisLayer*MiniBatchSize
                int Height = (Input.GetLength(0) - 1) * Stride + 1;//previousLayersHeight
                int Width = (Input.GetLength(1) - 1) * Stride + 1;//previousLayersWidth
                double[,,] D_ReStrided = new double[Height, Width, Input.GetLength(2)];//D_ReStrided is filled with Zeros
                for (int i = 0; i < Input.GetLength(0); i++)//inputs Height
                {
                    for (int j = 0; j < Input.GetLength(1); j++)//inputs width
                    {
                        for (int k = 0; k < Input.GetLength(2); k++)//inputs depth
                        {
                            D_ReStrided[i * Stride, j * Stride, k] = Input[i, j, k];
                        }
                    }
                }
                #region ZeroPadding
                int ZeroPadding = (FilterSize - 1) / 2;
                double[,,] InputZeroPadded = new double[D_ReStrided.GetLength(0) + 2 * ZeroPadding, D_ReStrided.GetLength(1) + 2 * ZeroPadding, D_ReStrided.GetLength(2)]; //InputZeroPadded is filled with Zeros
                for (int i = 0; i < D_ReStrided.GetLength(0); i++)
                {
                    for (int j = 0; j < D_ReStrided.GetLength(1); j++)
                    {
                        for (int k = 0; k < D_ReStrided.GetLength(2); k++)
                        {
                            InputZeroPadded[i + ZeroPadding, j + ZeroPadding, k] = D_ReStrided[i, j, k];
                        }
                    }
                }
                #endregion
                double[,] Output = new double[FilterSize * FilterSize * InputZeroPadded.GetLength(2) / MiniBatchSize, Height * Width * MiniBatchSize];//no Bias in Backprop || [ThisFilterSize*ThisFilterSize*ThisDepth, PreviousHeight*PreviousWidth*MiniBatchSize]          
                #region rearranging Input 
                //calculating Output
                int NewStride = 1; //rearrange the newly-formed Input (D_ReStrided) with a NewStride of 1
                for (int b = 0; b < MiniBatchSize; b++)
                {
                    for (int j = 0; j < Width; j++)
                    {
                        for (int i = 0; i < Height; i++)
                        {
                            for (int k = 0; k < InputZeroPadded.GetLength(2) / MiniBatchSize; k++)//DepthOfThisLayer=InputZeroPadded.GetLength(2)/MiniBatchSize
                            {
                                for (int jj = 0; jj < FilterSize; jj++)
                                {
                                    for (int ii = 0; ii < FilterSize; ii++)
                                    {
                                        Output[k * (FilterSize * FilterSize) + jj * FilterSize + ii, b * (Height * Width) + j * Height + i] = InputZeroPadded[i * NewStride + ii, j * NewStride + jj, k + b * InputZeroPadded.GetLength(2) / MiniBatchSize];
                                    }
                                }
                            }
                        }
                    }
                }
                #endregion
                return TransposeMatrix(Output);//Muss Zeropadding haben haben, sowie Transponiert werden. Kein Bias
            }
            #endregion
        }
        public static double[,] UnRotateWeightMat(double[,] NextLayersRotWeightMatNoBias, int NextLayersFilterSize)
        {
            int NextLayersFilterSizeSquared = NextLayersFilterSize * NextLayersFilterSize;
            int DepthOfPreviousLayer = NextLayersRotWeightMatNoBias.GetLength(0) / NextLayersFilterSizeSquared;//NextLayersRotWeightMatNoBias.GetLength(0) == FilterSizeSquared * DepthOfPreviousLayer
            double[,] NextLayersUnRotatedWeightMat = new double[NextLayersFilterSizeSquared * NextLayersRotWeightMatNoBias.GetLength(1), DepthOfPreviousLayer];// NextLayersRotWeightMatNoBias.GetLength(1) = Depth of next Layer
            for (int i = 0; i < NextLayersUnRotatedWeightMat.GetLength(1); i++)
            {
                for (int j = 0; j < NextLayersUnRotatedWeightMat.GetLength(0); j++)
                {
                    int SecondDim = j / NextLayersFilterSizeSquared;
                    double temp = NextLayersRotWeightMatNoBias[(i + 1) * NextLayersFilterSizeSquared - 1 - (j - SecondDim * NextLayersFilterSizeSquared), SecondDim];
                    NextLayersUnRotatedWeightMat[j, i] = temp;
                }
            }         
            return NextLayersUnRotatedWeightMat;
        }
        /// <summary>
        /// Computes Z_3D out of Z
        /// </summary>
        /// <param name="Z"></param>
        /// <param name="Z_3D"></param>
        /// <param name="MiniBatchSize"></param>
        /// <returns></returns>
        public static double[,,] ComputeZ_3D(double[,] Z, double[,,] Z_3D, int MiniBatchSize)//Z_3D[Height, Width, Depth*MiniBatchSize]
        {
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int k = 0; k < Z_3D.GetLength(2)/ MiniBatchSize; k++)//Depth
                {
                    for (int j = 0; j < Z_3D.GetLength(1); j++)//Width
                    {
                        for (int i = 0; i < Z_3D.GetLength(0); i++)//Height
                        {
                            Z_3D[i, j, b * Z_3D.GetLength(2) / MiniBatchSize + k] = Z[b, i + j * Z_3D.GetLength(0) + k * Z_3D.GetLength(0) * Z_3D.GetLength(1)];
                        }
                    }
                }
            }
            return Z_3D;
        }
        /// <summary>
        /// Computes Z out of Z_3D
        /// </summary>
        /// <param name="Z"></param>
        /// <param name="Z_3D"></param>
        /// <param name="MiniBatchSize"></param>
        /// <returns></returns>
        public static double[,] ComputeZ(double[,] Z, double[,,] Z_3D, int MiniBatchSize)//Z_3D[Height, Width, Depth*MiniBatchSize]
        {
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int k = 0; k < Z_3D.GetLength(2) / MiniBatchSize; k++)//Depth
                {
                    for (int j = 0; j < Z_3D.GetLength(1); j++)//Width
                    {
                        for (int i = 0; i < Z_3D.GetLength(0); i++)//Height
                        {
                            Z[b, i + j * Z_3D.GetLength(0) + k * Z_3D.GetLength(0) * Z_3D.GetLength(1)] = Z_3D[i, j, b * Z_3D.GetLength(2) / MiniBatchSize + k];
                        }
                    }
                }
            }
            return Z;
        }
        public static double[,,] ComputeD_3D(double[,] D, double[,,] D_3D, int MiniBatchSize)//D_3D[Height, Width, Depth*MiniBatchSize]
        {
            for (int b = 0; b < MiniBatchSize; b++)
            {
                for (int k = 0; k < D_3D.GetLength(2) / MiniBatchSize; k++)//Depth
                {
                    for (int j = 0; j < D_3D.GetLength(1); j++)//Width
                    {
                        for (int i = 0; i < D_3D.GetLength(0); i++)//Height
                        {
                            D_3D[i, j, b * D_3D.GetLength(2) / MiniBatchSize + k] = D[i + j * D_3D.GetLength(0) + b * D_3D.GetLength(0) * D_3D.GetLength(1), k];
                        }
                    }
                }
            }
            return D_3D;
        }
    }
}
