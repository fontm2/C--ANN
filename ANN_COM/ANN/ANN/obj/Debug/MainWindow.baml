   M S B A M L   `   `   `  ����>  :ANN, Version=1.0.0.0, Culture=neutral, PublicKeyToken=nullclr-namespace:ANNANN  5   	   m iSystem.Windows.Forms.DataVisualization, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35�mclr-namespace:System.Windows.Forms.DataVisualization.Charting;assembly=System.Windows.Forms.DataVisualization/System.Windows.Forms.DataVisualization.Charting 5   	       ANN.MainWindow   5   	   W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35^ ZWindowsFormsIntegration, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   #localclr-namespace:ANN     5   	   �winformchartmclr-namespace:System.Windows.Forms.DataVisualization.Charting;assembly=System.Windows.Forms.DataVisualization     5   	     � Title$  
MainWindow��5	   	   $	��450��6   $
��1500��6)   .��5
      �  � ColumnDefinitions 5   
   �� 5      $��1* �6   �� 5      $��1* �6   �� 5      $��1* �6   �� 5      $��3* �6   �� 5      $��3* �6    � RowDefinitions 5   
   �� 5      $��1* �6   �� 5      $��1* �6   �� 5      $��1* �6   .+�5   
   � $��0��6   $��0��6    5      �� 5      $��1* �6#    5      �� 5      $��1* �6    �� 5      $��1* �6    �� 5      $��1* �6    �� 5      $��1* �6    .+�5!      �� $��0��6   $��0��6$   $��Start New Net��61   6K   �� 5"      -   5        $��cb_StartNewNet��5"   4   $��0��6   $��1��6'    d� 	IsChecked$ FalseY�6J   6^   �� 5#      -   5        $��ErrorInformation��5#   1   $��0��6   $��2��6$   $
��Auto��6I   $��0,5,0,0q�6V   
���   �6g   � 5$      $��0��6   $��3��6$    5%      �� 5&      $��1* �6'   �� 5'      $��1* �6'    5)      �� 5*      $��1* �6$   .+�5,      �� $��0��6   $��0��6(   $��MNIST��65   6G   �� 5-      -   5        $��cb_MNIST��5-   8   $��1��6   $��0��6+   $
 trueY�6H   6[   5.      50      � 51   
   $��1��6   $��0��6    52      �� 53      $��1* �6#    55      �� 56      $��1* �6    �� 57      $��1* �6    �� 58      $��1* �6    .+�5:      �� -   5        $��ShowFeatureMapButton��5:   2   $��0��6   $��0��6%   $��Show Featuremaps��6N   6�   �� 5;      -   5        $��StartLearning��5;   2   $��0��6   $��1��6%   $��Start Learning��6G   6~   �� 5<      -   5        $��tbMinibatchSize��5<   =   $��0��6   $��2��6&   $��25��63   6V   5=      � 5>   
   $��2��6   $��0��6    5?      �� 5@      $��1* �6#    5B      �� 5C      $��1* �6    �� 5D      $��1* �6    �� 5E      $��1* �6    .+�5G      �� -   5        $��SaveWeights��5G   2   $��0��6   $��0��6%   $��SaveWeights��6E   6}   �� 5H      -   5        $��LoadWeights��5H   2   $��0��6   $��1��6%   $��LoadWeights��6E   6}   �� 5I      -	   5        $��tbSafeOffset��5I   <   $��0��6   $��2��6&   $��0��63   6R   5J      �� 5K   
   $��1��6   $��0��6"    �� Orientation$ Vertical=�63   
�������.+�5L      �� -
   5        $��camera1��5L      �� 5M      -   5        $��canvas1��5M      5N      �� 5O   
   $��1��6   $��1��6"   $ Vertical=�63   
�������.+�5P      �� -   5        $��camera2��5P      �� 5Q      -   5        $��canvas2��5Q      5R      �� 5S   
   $��1��6   $��2��6"   $ Vertical=�63   
�������.+�5T      �� -   5        $��camera3��5T      �� 5U      -   5        $��canvas3��5U      5V      �� 5W   
   $��2��6   $��0��6"   $ Vertical=�63   
�������.+�5X      �� -   5        $��camera4��5X      �� 5Y      -   5        $��canvas4��5Y      5Z      �� 5[   
   $��2��6   $��1��6"   $ Vertical=�63   
�������.+�5\      �� -   5        $��camera5��5\      �� 5]      -   5        $��canvas5��5]      5^      �� 5_   
   $��2��6   $��2��6"   $ Vertical=�63   
�������.+�5`      �� -   5        $��camera6��5`      �� 5a      -   5        $��canvas6��5a      5b      7  1System.Windows.Forms.Integration.WindowsFormsHost  5c   
   -   5          Name$
 host��5c      $��0��6)   $��3��66   $��3��6G      
Background
 �����$��1,1,1,1q�6j      Child. 5d      ;  5System.Windows.Forms.DataVisualization.Charting.Chart  -   5        StreamChart��[ WSystem.Windows.Forms, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089"  System.Windows.Forms.Control   Dock$
 Fill=�5d   6   	   Series	 5e      <  6System.Windows.Forms.DataVisualization.Charting.Series  5f      
   Name$
 series��6*      	ChartType$
 Line=�68      
ChartAreas 5h      ?  9System.Windows.Forms.DataVisualization.Charting.ChartArea  5i      5j      5l      5m        5n   
   -   5        $ host2��5n      $��0��6*   $��3��67   $��4��6H   
 �����$��1,1,1,1q�6k   . 5o        -   5        StreamChart2��$
 Fill=�5o   7   	 5p        5q      $
 series2��6*   $
 Line=�69    5s        5t      5u      5w      5x      5y      5z      