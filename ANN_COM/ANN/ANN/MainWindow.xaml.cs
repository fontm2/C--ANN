
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NueralNetwork;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms.DataVisualization.Charting;
using System.Drawing;

namespace ANN
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private int epoche = 0;
        private double minerr = 0.001;
        Network Net;
        private Task NetWorkTrainingTask;
        private string file;
        bool pause = false;
        string SaveWeightsPath;
        int SafeOffset;
        private Dictionary<Int64, double> value;
        private Chart chart;
        Int64 c = 0;
        private Dictionary<Int64, double> value2;
        private Chart chart2;
        Int64 d = 0;



        public MainWindow()
        {
            InitializeComponent();
            value = new Dictionary<Int64, double>();
            chart = this.FindName("StreamChart") as Chart;
            chart.DataSource = value;
            chart.Series["series"].XValueMember = "Key";
            chart.Series["series"].YValueMembers = "Value";
            chart.Series["series"].BorderWidth = 3;
            chart.Series["series"].BorderColor = System.Drawing.Color.Black;
            chart.Series["series"].Color = System.Drawing.Color.Black;
            //Enable Zoom
            chart.ChartAreas[0].AxisY.ScaleView.Zoomable = true;
            chart.ChartAreas[0].AxisX.ScaleView.Zoomable = true;
            chart.ChartAreas[0].CursorX.IsUserEnabled = true;
            chart.ChartAreas[0].CursorY.IsUserEnabled = true;
            chart.ChartAreas[0].CursorX.IsUserSelectionEnabled = true;
            chart.ChartAreas[0].CursorY.IsUserSelectionEnabled = true;

            value2 = new Dictionary<Int64, double>();
            chart2 = this.FindName("StreamChart2") as Chart;
            chart2.DataSource = value2;
            chart2.Series["series2"].XValueMember = "Key";
            chart2.Series["series2"].YValueMembers = "Value";
            chart2.Series["series2"].BorderWidth = 3;
            chart2.Series["series2"].BorderColor = System.Drawing.Color.Black;
            chart2.Series["series2"].Color = System.Drawing.Color.Black;
            //Enable Zoom
            chart2.ChartAreas[0].AxisY.ScaleView.Zoomable = true;
            chart2.ChartAreas[0].AxisX.ScaleView.Zoomable = true;
            chart2.ChartAreas[0].CursorX.IsUserEnabled = true;
            chart2.ChartAreas[0].CursorY.IsUserEnabled = true;
            chart2.ChartAreas[0].CursorX.IsUserSelectionEnabled = true;
            chart2.ChartAreas[0].CursorY.IsUserSelectionEnabled = true;

            ////MessageBox.Show(MLP[0].GetType().ToString() + "\n" + MLP[1].GetType().ToString() + "\n" + MLP[2].GetType().ToString() + "\n" + MLP[3].GetType().ToString() + "\n" + MLP[4].GetType().ToString());
            //string p1 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //string[] s = p1.Split(new[] { "bin" }, StringSplitOptions.None); //die Pattern sind unter dem aktuellen Projekt gespeichert, dort wo auch der Ordner "bin" ist.
            //file = s[0] + @"Pattern\NumbersPattern10Output4.txt";
            //Net = new Network();
            //Net.Load(file);
            //NetWorkTrainingTask = Task.Factory.StartNew(() => TrainNetwork());


            #region Learning Colors
            //string p1 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //string[] s = p1.Split(new[] { "bin" }, StringSplitOptions.None); //die Pattern sind unter dem aktuellen Projekt gespeichert, dort wo auch der Ordner "bin" ist.
            //string[] colorinputsdata = new string[15];
            //colorinputsdata[0] = s[0] + @"Color\red1.png";
            //colorinputsdata[1] = s[0] + @"Color\green1.png";
            //colorinputsdata[2] = s[0] + @"Color\blue1.png";
            //colorinputsdata[3] = s[0] + @"Color\red2.png";
            //colorinputsdata[4] = s[0] + @"Color\green2.png";
            //colorinputsdata[5] = s[0] + @"Color\blue2.png";
            //colorinputsdata[6] = s[0] + @"Color\red3.png";
            //colorinputsdata[7] = s[0] + @"Color\green3.png";
            //colorinputsdata[8] = s[0] + @"Color\blue3.png";
            //colorinputsdata[9] = s[0] + @"Color\red4.png";
            //colorinputsdata[10] = s[0] + @"Color\green4.png";
            //colorinputsdata[11] = s[0] + @"Color\blue4.png";
            //colorinputsdata[12] = s[0] + @"Color\red5.png";
            //colorinputsdata[13] = s[0] + @"Color\green5.png";
            //colorinputsdata[14] = s[0] + @"Color\blue5.png";
            //double[,] Outputs = new double[,] { { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 } };
            //Net = new Network(colorinputsdata, Outputs);
            //NetWorkTrainingTask = Task.Factory.StartNew(() => TrainNetwork());
            #endregion
            #region Shapes
            //string p1 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //string[] s = p1.Split(new[] { "bin" }, StringSplitOptions.None); //die Pattern sind unter dem aktuellen Projekt gespeichert, dort wo auch der Ordner "bin" ist.
            //#region Longshapes
            ////string[] Shapeinputsdata = new string[66];
            ////Shapeinputsdata[0] = s[0] + @"Shapes\circle1.png";
            ////Shapeinputsdata[1] = s[0] + @"Shapes\square1.png";
            ////Shapeinputsdata[2] = s[0] + @"Shapes\cross1.png";
            ////Shapeinputsdata[3] = s[0] + @"Shapes\circle2.png";
            ////Shapeinputsdata[4] = s[0] + @"Shapes\square2.png";
            ////Shapeinputsdata[5] = s[0] + @"Shapes\cross2.png";
            ////Shapeinputsdata[6] = s[0] + @"Shapes\circle3.png";
            ////Shapeinputsdata[7] = s[0] + @"Shapes\square3.png";
            ////Shapeinputsdata[8] = s[0] + @"Shapes\cross3.png";
            ////Shapeinputsdata[9] = s[0] + @"Shapes\circle4.png";
            ////Shapeinputsdata[10] = s[0] + @"Shapes\square4.png";
            ////Shapeinputsdata[11] = s[0] + @"Shapes\cross4.png";
            ////Shapeinputsdata[12] = s[0] + @"Shapes\circle5.png";
            ////Shapeinputsdata[13] = s[0] + @"Shapes\square5.png";
            ////Shapeinputsdata[14] = s[0] + @"Shapes\cross5.png";
            ////Shapeinputsdata[15] = s[0] + @"Shapes\circle6.png";
            ////Shapeinputsdata[16] = s[0] + @"Shapes\square6.png";
            ////Shapeinputsdata[17] = s[0] + @"Shapes\cross6.png";
            ////Shapeinputsdata[18] = s[0] + @"Shapes\circle7.png";
            ////Shapeinputsdata[19] = s[0] + @"Shapes\square7.png";
            ////Shapeinputsdata[20] = s[0] + @"Shapes\cross7.png";
            ////Shapeinputsdata[21] = s[0] + @"Shapes\circle8.png";
            ////Shapeinputsdata[22] = s[0] + @"Shapes\square8.png";
            ////Shapeinputsdata[23] = s[0] + @"Shapes\cross8.png";
            ////Shapeinputsdata[24] = s[0] + @"Shapes\circle9.png";
            ////Shapeinputsdata[25] = s[0] + @"Shapes\square9.png";
            ////Shapeinputsdata[26] = s[0] + @"Shapes\cross9.png";
            ////Shapeinputsdata[27] = s[0] + @"Shapes\circle10.png";
            ////Shapeinputsdata[28] = s[0] + @"Shapes\square10.png";
            ////Shapeinputsdata[29] = s[0] + @"Shapes\cross10.png";
            ////Shapeinputsdata[30] = s[0] + @"Shapes\circle11.png";
            ////Shapeinputsdata[31] = s[0] + @"Shapes\square11.png";
            ////Shapeinputsdata[32] = s[0] + @"Shapes\cross11.png";

            ////Shapeinputsdata[33] = s[0] + @"Shapes\circle12.png";
            ////Shapeinputsdata[34] = s[0] + @"Shapes\square12.png";
            ////Shapeinputsdata[35] = s[0] + @"Shapes\cross12.png";
            ////Shapeinputsdata[36] = s[0] + @"Shapes\circle13.png";
            ////Shapeinputsdata[37] = s[0] + @"Shapes\square13.png";
            ////Shapeinputsdata[38] = s[0] + @"Shapes\cross13.png";
            ////Shapeinputsdata[39] = s[0] + @"Shapes\circle14.png";
            ////Shapeinputsdata[40] = s[0] + @"Shapes\square14.png";
            ////Shapeinputsdata[41] = s[0] + @"Shapes\cross14.png";
            ////Shapeinputsdata[42] = s[0] + @"Shapes\circle15.png";
            ////Shapeinputsdata[43] = s[0] + @"Shapes\square15.png";
            ////Shapeinputsdata[44] = s[0] + @"Shapes\cross15.png";
            ////Shapeinputsdata[45] = s[0] + @"Shapes\circle16.png";
            ////Shapeinputsdata[46] = s[0] + @"Shapes\square16.png";
            ////Shapeinputsdata[47] = s[0] + @"Shapes\cross16.png";
            ////Shapeinputsdata[48] = s[0] + @"Shapes\circle17.png";
            ////Shapeinputsdata[49] = s[0] + @"Shapes\square17.png";
            ////Shapeinputsdata[50] = s[0] + @"Shapes\cross17.png";
            ////Shapeinputsdata[51] = s[0] + @"Shapes\circle18.png";
            ////Shapeinputsdata[52] = s[0] + @"Shapes\square18.png";
            ////Shapeinputsdata[53] = s[0] + @"Shapes\cross18.png";
            ////Shapeinputsdata[54] = s[0] + @"Shapes\circle19.png";
            ////Shapeinputsdata[55] = s[0] + @"Shapes\square19.png";
            ////Shapeinputsdata[56] = s[0] + @"Shapes\cross19.png";
            ////Shapeinputsdata[57] = s[0] + @"Shapes\circle20.png";
            ////Shapeinputsdata[58] = s[0] + @"Shapes\square20.png";
            ////Shapeinputsdata[59] = s[0] + @"Shapes\cross20.png";
            ////Shapeinputsdata[60] = s[0] + @"Shapes\circle21.png";
            ////Shapeinputsdata[61] = s[0] + @"Shapes\square21.png";
            ////Shapeinputsdata[62] = s[0] + @"Shapes\cross21.png";
            ////Shapeinputsdata[63] = s[0] + @"Shapes\circle22.png";
            ////Shapeinputsdata[64] = s[0] + @"Shapes\square22.png";
            ////Shapeinputsdata[65] = s[0] + @"Shapes\cross22.png";
            ////double[,] Outputs = new double[,]
            ////{
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },

            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            ////    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }
            ////};
            //#endregion
            //#region shortshapes
            //string[] Shapeinputsdata = new string[33];
            //Shapeinputsdata[0] = s[0] + @"Shapes\circle1.png";
            //Shapeinputsdata[1] = s[0] + @"Shapes\square1.png";
            //Shapeinputsdata[2] = s[0] + @"Shapes\cross1.png";
            //Shapeinputsdata[3] = s[0] + @"Shapes\circle2.png";
            //Shapeinputsdata[4] = s[0] + @"Shapes\square2.png";
            //Shapeinputsdata[5] = s[0] + @"Shapes\cross2.png";
            //Shapeinputsdata[6] = s[0] + @"Shapes\circle3.png";
            //Shapeinputsdata[7] = s[0] + @"Shapes\square3.png";
            //Shapeinputsdata[8] = s[0] + @"Shapes\cross3.png";
            //Shapeinputsdata[9] = s[0] + @"Shapes\circle4.png";
            //Shapeinputsdata[10] = s[0] + @"Shapes\square4.png";
            //Shapeinputsdata[11] = s[0] + @"Shapes\cross4.png";
            //Shapeinputsdata[12] = s[0] + @"Shapes\circle5.png";
            //Shapeinputsdata[13] = s[0] + @"Shapes\square5.png";
            //Shapeinputsdata[14] = s[0] + @"Shapes\cross5.png";
            //Shapeinputsdata[15] = s[0] + @"Shapes\circle6.png";
            //Shapeinputsdata[16] = s[0] + @"Shapes\square6.png";
            //Shapeinputsdata[17] = s[0] + @"Shapes\cross6.png";
            //Shapeinputsdata[18] = s[0] + @"Shapes\circle7.png";
            //Shapeinputsdata[19] = s[0] + @"Shapes\square7.png";
            //Shapeinputsdata[20] = s[0] + @"Shapes\cross7.png";
            //Shapeinputsdata[21] = s[0] + @"Shapes\circle8.png";
            //Shapeinputsdata[22] = s[0] + @"Shapes\square8.png";
            //Shapeinputsdata[23] = s[0] + @"Shapes\cross8.png";
            //Shapeinputsdata[24] = s[0] + @"Shapes\circle9.png";
            //Shapeinputsdata[25] = s[0] + @"Shapes\square9.png";
            //Shapeinputsdata[26] = s[0] + @"Shapes\cross9.png";
            //Shapeinputsdata[27] = s[0] + @"Shapes\circle10.png";
            //Shapeinputsdata[28] = s[0] + @"Shapes\square10.png";
            //Shapeinputsdata[29] = s[0] + @"Shapes\cross10.png";
            //Shapeinputsdata[30] = s[0] + @"Shapes\circle11.png";
            //Shapeinputsdata[31] = s[0] + @"Shapes\square11.png";
            //Shapeinputsdata[32] = s[0] + @"Shapes\cross11.png";

            //double[,] Outputs = new double[,]
            //{
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 },
            //    { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }
            //};
            //#endregion
            //Net = new Network(Shapeinputsdata, Outputs);
            //NetWorkTrainingTask = Task.Factory.StartNew(() => TrainNetwork());
            #endregion


            #region Images
            //string p1 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //string[] s = p1.Split(new[] { "bin" }, StringSplitOptions.None); //die Pattern sind unter dem aktuellen Projekt gespeichert, dort wo auch der Ordner "bin" ist.
            //string fileName = s[0] + @"\ImagesAI\data_batch_2.bin";
            //Net = new Network(fileName,200);
            //NetWorkTrainingTask = Task.Factory.StartNew(() => TrainNetwork());          
            #endregion
        }
        private void TrainNetwork()
        {
            double error = 0;
            do
            {
                if (!pause)
                {
                    error = Net.Train();
                    epoche++;
                    //calls the UI-Thread for Updateing the Labels Content and ProgressBars Value
                    Dispatcher.Invoke(new Action(() =>
                    {
                        ErrorInformation.Content = "Iteration: " + epoche.ToString() + " | Error: " + Math.Round(error, 5).ToString();
                    }));                  
                }
            }
            while (error > minerr);
        }

        private void ShowFeatureMapButton_Click(object sender, RoutedEventArgs e)
        {            
            pause = !pause;
            WriteableBitmap image1 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            double[] temp = new double[(Net[1].Z_3D.GetLength(0) * image1.BackBufferStride)];
            double smallestValue = 0;
            double biggestValue = 0;
            //byte[] imageByteArray = new byte[(Net[1].Z_3D.GetLength(0) * Net[1].Z_3D.GetLength(1))];//[Height*Width]
            //for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            //{
            //    for (int j = 0; j < Net[1].Z_3D.GetLength(1); j++)
            //    {
            //        imageByteArray[j + i * Net[1].Z_3D.GetLength(1)] = Convert.ToByte(Net[1].Z_3D[i, j,0] * 100);
            //    }
            //}
            //image1.Lock();
            //Marshal.Copy(imageByteArray, 0, image1.BackBuffer, imageByteArray.Length);
            //image1.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            //image1.Unlock();
            byte[] imageByteArray = new byte[(Net[1].Z_3D.GetLength(0) * image1.BackBufferStride)];//[Height*BackBufferStride]
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {
                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if(Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];                    
                    }                                     
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue))/(biggestValue + Math.Abs(smallestValue))*255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image1.Lock();
            Marshal.Copy(imageByteArray, 0, image1.BackBuffer, imageByteArray.Length);
            image1.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image1.Unlock();
            camera1.Source = image1;
            canvas1.Height = image1.PixelHeight * 4;
            canvas1.Width = image1.PixelWidth * 4;
            camera1.Height = image1.PixelHeight * 4;
            camera1.Width = image1.PixelWidth * 4;

            WriteableBitmap image2 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {

                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if (Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];
                    }
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue)) / (biggestValue + Math.Abs(smallestValue)) * 255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image2.Lock();
            Marshal.Copy(imageByteArray, 0, image2.BackBuffer, imageByteArray.Length);
            image2.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image2.Unlock();
            camera2.Source = image2;
            canvas2.Height = image2.PixelHeight * 4;
            canvas2.Width = image2.PixelWidth * 4;
            camera2.Height = image2.PixelHeight * 4;
            camera2.Width = image2.PixelWidth * 4;

            WriteableBitmap image3 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {

                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if (Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];
                    }
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue)) / (biggestValue + Math.Abs(smallestValue)) * 255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image3.Lock();
            Marshal.Copy(imageByteArray, 0, image3.BackBuffer, imageByteArray.Length);
            image3.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image3.Unlock();
            camera3.Source = image3;
            canvas3.Height = image3.PixelHeight * 4;
            canvas3.Width = image3.PixelWidth * 4;
            camera3.Height = image3.PixelHeight * 4;
            camera3.Width = image3.PixelWidth * 4;

            WriteableBitmap image4 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {

                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if (Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];
                    }
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue)) / (biggestValue + Math.Abs(smallestValue)) * 255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image4.Lock();
            Marshal.Copy(imageByteArray, 0, image4.BackBuffer, imageByteArray.Length);
            image4.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image4.Unlock();
            camera4.Source = image4;
            canvas4.Height = image4.PixelHeight * 4;
            canvas4.Width = image4.PixelWidth * 4;
            camera4.Height = image4.PixelHeight * 4;
            camera4.Width = image4.PixelWidth * 4;

            WriteableBitmap image5 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {

                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if (Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];
                    }
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue)) / (biggestValue + Math.Abs(smallestValue)) * 255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image5.Lock();
            Marshal.Copy(imageByteArray, 0, image5.BackBuffer, imageByteArray.Length);
            image5.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image5.Unlock();
            camera5.Source = image5;
            canvas5.Height = image5.PixelHeight * 4;
            canvas5.Width = image5.PixelWidth * 4;
            camera5.Height = image5.PixelHeight * 4;
            camera5.Width = image5.PixelWidth * 4;

            WriteableBitmap image6 = new WriteableBitmap(Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1), 96.0, 96.0, PixelFormats.Gray8, null);
            for (int i = 0; i < Net[1].Z_3D.GetLength(0); i++)
            {
                for (int j = 0; j < image1.BackBufferStride; j++)
                {
                    if (j == Net[1].Z_3D.GetLength(1))
                    {
                        imageByteArray[j + i * image1.BackBufferStride] = 255;
                    }
                    else
                    {

                        if (Net[1].Z_3D[i, j, 0] < smallestValue)
                        {
                            smallestValue = Net[1].Z_3D[i, j, 0];
                        }
                        if (Net[1].Z_3D[i, j, 0] > biggestValue)
                        {
                            biggestValue = Net[1].Z_3D[i, j, 0];
                        }
                        temp[j + i * image1.BackBufferStride] = Net[1].Z_3D[i, j, 0];
                    }
                }
            }
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (temp[i] + Math.Abs(smallestValue)) / (biggestValue + Math.Abs(smallestValue)) * 255;
                imageByteArray[i] = Convert.ToByte(temp[i]);
            }
            image6.Lock();
            Marshal.Copy(imageByteArray, 0, image6.BackBuffer, imageByteArray.Length);
            image6.AddDirtyRect(new Int32Rect(0, 0, Net[1].Z_3D.GetLength(0), Net[1].Z_3D.GetLength(1)));
            image6.Unlock();
            camera6.Source = image6;
            canvas6.Height = image6.PixelHeight * 4;
            canvas6.Width = image6.PixelWidth * 4;
            camera6.Height = image6.PixelHeight * 4;
            camera6.Width = image6.PixelWidth * 4;

        }

        private void SaveWeightsButton_Click(object sender, RoutedEventArgs e)
        {
            string path = @"D:\Weights_Traine1MioEpoches.txt";
            {
                // Create a file to write to.
                using (StreamWriter sw = File.CreateText(path))
                {
                    for (int k = 1; k < Net.Count; k++)//k=1, because InputLayer has no Weights
                    {                      
                        for (int i = 0; i < Net[k].W.GetLength(0); i++)
                        {
                            for (int j = 0; j < Net[k].W.GetLength(1); j++)
                            {
                                sw.Write(Net[k].W[i,j].ToString());
                                sw.Write(";");
                            }
                            sw.WriteLine();
                        }
                    }
                }
            }
        }

        private void LoadWeightsButton_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.DefaultExt = ".bin";
            dlg.Filter = "Text documents (.txt)|*.txt|" + "All Files|*.*";
            Nullable<bool> result = dlg.ShowDialog();
            if (result == true)
            {
                string path = dlg.FileName;
                string line;
                string[] split;
                {
                    // Create a file to write to.
                    using (StreamReader sr = new StreamReader(path))
                    {

                        for (int k = 1; k < Net.Count; k++)//k=1, because InputLayer has no Weights
                        {
                            for (int i = 0; i < Net[k].W.GetLength(0); i++)
                            {
                                line = sr.ReadLine();
                                split = line.Split(';');
                                for (int j = 0; j < Net[k].W.GetLength(1); j++)
                                {
                                    Net[k].W[i, j] = Convert.ToDouble(split[j]);
                                }
                            }
                        }
                    }
                }
            }
        }

        private void StartLearning_Click(object sender, RoutedEventArgs e)
        {
            SafeOffset = int.Parse(tbSafeOffset.Text);
            string p1 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            string[] s = p1.Split(new[] { "ANN_COM" }, StringSplitOptions.None); //die Pattern sind unter dem aktuellen Projekt gespeichert, dort wo auch der Ordner "bin" ist.
            if (cb_MNIST.IsChecked == false)
            {
                string[] ListOfFilenames = new string[6];
                ListOfFilenames[0] = s[0] + @"\ANN_COM\ImagesAI\data_batch_1.bin";
                ListOfFilenames[1] = s[0] + @"\ANN_COM\ImagesAI\data_batch_2.bin";
                ListOfFilenames[2] = s[0] + @"\ANN_COM\ImagesAI\data_batch_3.bin";
                ListOfFilenames[3] = s[0] + @"\ANN_COM\ImagesAI\data_batch_4.bin";
                ListOfFilenames[4] = s[0] + @"\ANN_COM\ImagesAI\data_batch_5.bin";
                ListOfFilenames[5] = s[0] + @"\ANN_COM\ImagesAI\test_batch.bin";
                Net = new Network(ListOfFilenames[0], int.Parse(tbMinibatchSize.Text), ListOfFilenames, false);
            }
            else
            {
                string[] ListOfMNISTFilenames = new string[4];
                ListOfMNISTFilenames[0] = s[0] + @"\ANN_COM\MNIST-Data\train-images.idx3-ubyte";
                ListOfMNISTFilenames[1] = s[0] + @"\ANN_COM\MNIST-Data\train-labels.idx1-ubyte";
                ListOfMNISTFilenames[2] = s[0] + @"\ANN_COM\MNIST-Data\t10k-images.idx3-ubyte";
                ListOfMNISTFilenames[3] = s[0] + @"\ANN_COM\MNIST-Data\t10k-labels.idx1-ubyte";
                Net = new Network(ListOfMNISTFilenames[0], int.Parse(tbMinibatchSize.Text), ListOfMNISTFilenames, true);
            }


            //Load weights
            if (cb_StartNewNet.IsChecked == false)
            {
                Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
                dlg.DefaultExt = ".bin";
                dlg.Filter = "Text documents (.txt)|*.txt|" + "All Files|*.*";
                Nullable<bool> result = dlg.ShowDialog();
                if (result == true)
                {
                    string path = dlg.FileName;
                    string line;
                    string[] split;
                    {
                        // Create a file to write to.
                        using (StreamReader sr = new StreamReader(path))
                        {

                            for (int k = 1; k < Net.Count; k++)//k=1, because InputLayer has no Weights
                            {
                                for (int i = 0; i < Net[k].W.GetLength(0); i++)
                                {
                                    line = sr.ReadLine();
                                    split = line.Split(';');
                                    for (int j = 0; j < Net[k].W.GetLength(1); j++)
                                    {
                                        split[j] = split[j].Replace('.', ',');
                                        Net[k].W[i, j] = Convert.ToDouble(split[j]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Microsoft.Win32.SaveFileDialog Sdlg = new Microsoft.Win32.SaveFileDialog();
            Sdlg.FileName = "ListOfWeights_AutomTrain_";           
            // Show save file dialog box
            Nullable<bool> save_result = Sdlg.ShowDialog();
            if (save_result == true)
            {
                // Save document 
                SaveWeightsPath = Sdlg.FileName;              
            }
            Net.SaveWeightsEvent += SaveWeightsEventEventHandler;
            Net.BadgeErrorEvent += BadgeErrorEventEventHandler;
            Net.NetAccuracyEvent += NetAccuracyEventEventHandler;
            Net.TestNetwork();//get initial random accuracy
            NetWorkTrainingTask = Task.Factory.StartNew(() => TrainNetwork());
        }

        private void BadgeErrorEventEventHandler(Network Net, double accumulatedError)
        {
            value.Add(c, accumulatedError);
            Dispatcher.Invoke(new Action(() =>
            {
                chart.DataBind();
            }));
            c++;
        }

        private void NetAccuracyEventEventHandler(Network Net, double accuracy)
        {
            value2.Add(d, accuracy);
            Dispatcher.Invoke(new Action(() =>
            {
                chart2.DataBind();
            }));
            d++;
        }
        public void SaveWeightsEventEventHandler(object sender)
        {
            pause = true;
            string path = SaveWeightsPath + (epoche + SafeOffset).ToString() + ".txt";
            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(path))
            {
                for (int k = 1; k < Net.Count; k++)//k=1, because InputLayer has no Weights
                {
                    for (int i = 0; i < Net[k].W.GetLength(0); i++)
                    {
                        for (int j = 0; j < Net[k].W.GetLength(1); j++)
                        {
                            sw.Write(Net[k].W[i, j].ToString());
                            sw.Write(";");
                        }
                        sw.WriteLine();
                    }
                }
            }
            pause = false;
        }
    }
}
