using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ImageLoader
{
    public static class ImageHandler
    {
        /// <summary>
        /// Returns a Matrix [Height, Width, MiniBatchSize*Depth]
        /// </summary>
        /// <param name="InputFileNames"></param>
        /// <returns></returns>
        public static double[,,]  ImageToZ_3D(string[] InputFileNames)
        {
            double[,,] temp;
            int MiniBatchSize = InputFileNames.Length;
            WriteableBitmap TempImage = LoadImageToWriteableBitmap(InputFileNames[0]);
            double[,,] Result = new double[TempImage.PixelHeight, TempImage.PixelWidth, MiniBatchSize * BytesPerPixel(TempImage)];
            for (int l = 0; l < InputFileNames.Length; l++)
            {
                TempImage = LoadImageToWriteableBitmap(InputFileNames[l]);
                temp = SplitImageToColors(TempImage);//[TempImage.PixelHeight, TempImage.PixelWidth, BytesPerPixel(TempImage)]
                for (int k = 0; k < temp.GetLength(2); k++)
                {
                    for (int i = 0; i < temp.GetLength(0); i++)
                    {
                        for (int j = 0; j < temp.GetLength(1); j++)
                        {                           
                            Result[i, j, l * temp.GetLength(2) + k] = temp[i, j, k];
                        }
                    }
                }
            }
            return Result;
        }
        private static WriteableBitmap LoadImageToWriteableBitmap(string filename)
        {
            BitmapImage bitmap_image = new BitmapImage(new Uri(filename, UriKind.Relative));
            WriteableBitmap _WriteableBitmap = new WriteableBitmap(bitmap_image);
            return _WriteableBitmap;
        }
        private static byte[] ConvertBitmapToByteArray(WriteableBitmap _WriteableBitmap)
        {
            int stride = Stride(_WriteableBitmap);

            byte[] bitmapData = new byte[_WriteableBitmap.PixelHeight * stride];

            _WriteableBitmap.CopyPixels(bitmapData, stride, 0);
            return bitmapData;
        }
        private static int Stride(WriteableBitmap _WriteableBitmap)
        {
            return _WriteableBitmap.PixelWidth * BytesPerPixel(_WriteableBitmap);
        }
        private static int BitsPerPixel(WriteableBitmap _WriteableBitmap)
        {
            return _WriteableBitmap.Format.BitsPerPixel;
        }
        private static int BytesPerPixel(WriteableBitmap _WriteableBitmap)
        {
            return (_WriteableBitmap.Format.BitsPerPixel + 7) / 8;
        }

        private static double[,,] SplitImageToColors(WriteableBitmap _WriteableBitmap)
        {
            byte[] inputs = ConvertBitmapToByteArray(_WriteableBitmap);
            double[,,] ColorSplit;
            switch (BytesPerPixel(_WriteableBitmap))
            {
                case 1:
                    ColorSplit = new double[_WriteableBitmap.PixelHeight, _WriteableBitmap.PixelWidth, 1];
                    for (int i = 0; i < _WriteableBitmap.PixelHeight; i++)
                    {
                        for (int j = 0; j < _WriteableBitmap.PixelWidth; j++)
                        {
                            ColorSplit[i, j, 0] = (double)inputs[i * _WriteableBitmap.PixelWidth + j] / 255;// Inputs are Normalized
                        }
                    }
                    break;
                case 2:
                    ColorSplit = new double[_WriteableBitmap.PixelHeight, _WriteableBitmap.PixelWidth, 2];
                    for (int k = 0; k < ColorSplit.GetLength(2); k++)//Image Depth
                    {
                        for (int i = 0; i < ColorSplit.GetLength(0); i++)//WriteableBitmap.PixelHeight
                        {
                            for (int j = 0; j < ColorSplit.GetLength(1); j++)//WriteableBitmap.PixelWidth
                            {
                                ColorSplit[i, j, k] = (double)inputs[i * ColorSplit.GetLength(1) * 2 + j * 2 + k] / 255;// Inputs are Normalized
                            }
                        }
                    }
                    break;
                case 3:
                    ColorSplit = new double[_WriteableBitmap.PixelHeight, _WriteableBitmap.PixelWidth, 3];
                    for (int k = 0; k < ColorSplit.GetLength(2); k++)//Image Depth
                    {
                        for (int i = 0; i < ColorSplit.GetLength(0); i++)//WriteableBitmap.PixelHeight
                        {
                            for (int j = 0; j < ColorSplit.GetLength(1); j++)//WriteableBitmap.PixelWidth
                            {
                                ColorSplit[i, j, k] = (double)inputs[i * ColorSplit.GetLength(1) * 3 + j * 3 + k] / 255;// Inputs are Normalized
                            }
                        }
                    }
                    break;
                case 4:
                    ColorSplit = new double[_WriteableBitmap.PixelHeight, _WriteableBitmap.PixelWidth, 4];
                    for (int k = 0; k < ColorSplit.GetLength(2); k++)//Image Depth
                    {
                        for (int i = 0; i < ColorSplit.GetLength(0); i++)//WriteableBitmap.PixelHeight
                        {
                            for (int j = 0; j < ColorSplit.GetLength(1); j++)//WriteableBitmap.PixelWidth
                            {
                                ColorSplit[i, j, k] = (double)inputs[i * ColorSplit.GetLength(1) * 4 + j * 4 + k] / 255;// Inputs are Normalized
                            }
                        }
                    }
                    break;
                default:
                ColorSplit = new double[0, 0, 0];
                break;
            }
            return ColorSplit;
        }
    }
}

