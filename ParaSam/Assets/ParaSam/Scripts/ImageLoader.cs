using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine;
using UnityEngine.UI;
using Rect = OpenCVForUnity.CoreModule.Rect;
using System.Linq;
using System.Text.RegularExpressions;

[System.Serializable]
public struct TumorConfig
{
    public string prefix;
    public string day;
    public string tumorFolder;
    public int startInclude;
    public int endInclude;
}

public class ImageLoader : MonoBehaviour
{
    [Header("Tumor Selector")] 
    public int activeTumor;
    // 0: tumorFolder 1:startInclude 2:endInclude
    public TumorConfig[] tumorConfig;
    [Header("Preprocess")]
    public string annotationFolderPath;
    public string originalTiffFolderPath;

    public bool cropBigImage;

    public string newAnnotationFolderPath;
    public string newOriginalTiffFolderPath;
    public string tumorFolder;
    public bool ready;
    
    public List<Texture2D> images;
    public List<Texture2D> blueShapesT;
    public List<Texture2D> blueShapes2T;
    private List<Mat> blueShapes2;
    private List<Mat> blueShapes3;
    public List<Texture2D> squareT;
    public List<Texture2D> square2T;
    public List<Texture2D> imagesCrop;
    public List<Texture2D> allContT;

    public Material uiMat;
    public RawImage rawImage;

    public int currentImage;

    public Visualize viz;

    public bool drawRect;

    [Header("Generate Folders")] 
    public string prefix;
    public bool excludeNoAnnotation;
    public int startInclude;
    public int endInclude;
    public string saveFolderPath;
    public string saveImagesFolderPath;

    public string newSaveFolderPath;
    public string newSaveImagesFolderPath;
    
    public string[] filePaths;
    public string[] originalFilePaths;
    public string[] filteredFiles;

    [Header("Generate Training Set")] 
    public bool resizeTo256;
    public float trainingSplit;
    public string trainFolderPath;
    public string testFolderPath;

    public string newTrainFolderPath;
    public string newTestFolderPath;
    
    public int[] indices;
    public string[] files;
    public string[] files2;
    public enum Visualize
    {
        IMAGE,
        THRESHOLD,
        SQUARE_DET,
        IMAGE_CROP,
        LABEL_DET,
        ALL_CONTOURS,
    }
    
    int CompareFilesBySuffix(string a, string b)
    {
        try
        {
            int suffixA = int.Parse(Regex.Match(Path.GetFileName(a), @"_(\d+)\.tif$").Groups[1].Value);
            int suffixB = int.Parse(Regex.Match(Path.GetFileName(b), @"_(\d+)\.tif$").Groups[1].Value);
            return suffixA.CompareTo(suffixB);
        }
        catch (FormatException ex)
        {
            // Log the error and the filenames that caused it for debugging purposes
            Console.WriteLine("An error occurred while parsing the filenames '{0}' and '{1}': {2}", Path.GetFileName(a),
                Path.GetFileName(b), ex.Message);
            // Handle the error (e.g., by returning 0, which will keep the original order)
            return 0;
        }
    }

    static void LogAsciiValues(string str)
    {
        foreach (char c in str)
        {
            Debug.Log(string.Format("Character: {0}, ASCII: {1}", c, (int)c));
        }
    }
    static string SanitizeFilename(string filename)
    {
        return Regex.Replace(filename, @"[^\d\(\).png]", "");
    }
    int CompareFilesBySuffixPNG(string a, string b)
    {
        try
        {
            string filenameA = Regex.Replace(Path.GetFileName(a), @"\s+", "");
            string filenameB = Regex.Replace(Path.GetFileName(b), @"\s+", "");

            filenameA = SanitizeFilename(filenameA);
            filenameB = SanitizeFilename(filenameB);
            //Debug.Log(filenameA + " " + filenameB);
            //LogAsciiValues(filenameA);
            //LogAsciiValues(filenameB);
            int suffixA = int.Parse(Regex.Match(filenameA, @"\((\d+)\)$").Groups[1].Value);
            int suffixB = int.Parse(Regex.Match(filenameB, @"\((\d+)\)$").Groups[1].Value);
            return suffixA.CompareTo(suffixB);
        }
        catch (FormatException ex)
        {
            // Log the error and the filenames that caused it for debugging purposes
            Debug.LogErrorFormat("An error occurred while parsing the filenames '{0}' and '{1}': {2}", Path.GetFileName(a),
                Path.GetFileName(b), ex.Message);
            // Handle the error (e.g., by returning 0, which will keep the original order)
            return 0;
        }
    }

    int CompareFilesBySuffixPNG2(string a, string b)
    {
        try
        {
            string filenameA = Regex.Replace(Path.GetFileName(a), @"\s+", "");
            string filenameB = Regex.Replace(Path.GetFileName(b), @"\s+", "");

            filenameA = SanitizeFilename(filenameA);
            filenameB = SanitizeFilename(filenameB);
            Debug.Log(filenameA + " " + filenameB);
            //LogAsciiValues(filenameA);
            //LogAsciiValues(filenameB);
            int suffixA = int.Parse(Regex.Match(filenameA, @"\((\d+)\).png$").Groups[1].Value);
            int suffixB = int.Parse(Regex.Match(filenameB, @"\((\d+)\).png$").Groups[1].Value);
            return suffixA.CompareTo(suffixB);
        }
        catch (FormatException ex)
        {
            // Log the error and the filenames that caused it for debugging purposes
            Debug.LogErrorFormat("An error occurred while parsing the filenames '{0}' and '{1}': {2}", Path.GetFileName(a),
                Path.GetFileName(b), ex.Message);
            // Handle the error (e.g., by returning 0, which will keep the original order)
            return 0;
        }
    }

    void Start()
    {
        activeTumor = 0;
        /*ready = false;
        activeTumor = 0;
        prefix = tumorConfig[activeTumor].prefix;
        tumorConfig[activeTumor].tumorFolder = tumorConfig[activeTumor].prefix + '_' + tumorConfig[activeTumor].day;
        tumorFolder = tumorConfig[activeTumor].tumorFolder;
        startInclude = tumorConfig[activeTumor].startInclude;
        endInclude = tumorConfig[activeTumor].endInclude;
        
        Preprocess();
        ready = true;*/
    }
    // Start is called before the first frame update
    public void Preprocess()
    {
        ready = false;
        prefix = tumorConfig[activeTumor].prefix;
        tumorConfig[activeTumor].tumorFolder = tumorConfig[activeTumor].prefix + '_' + tumorConfig[activeTumor].day;
        tumorFolder = tumorConfig[activeTumor].tumorFolder;
        startInclude = tumorConfig[activeTumor].startInclude;
        endInclude = tumorConfig[activeTumor].endInclude;
        
        images = new List<Texture2D>();
        blueShapesT = new List<Texture2D>();
        blueShapes2T = new List<Texture2D>();
        squareT = new List<Texture2D>();
        imagesCrop = new List<Texture2D>();
        allContT = new List<Texture2D>();
        
        //folderPath = "E:\\Ultrasound\\Unity\\Imgs\\KPC2604_0";
        //saveFolderPath = "E:\\Ultrasound\\Unity\\PreOutput\\KPC2604_0";
        
        filePaths = Directory.GetFiles(annotationFolderPath, "*.jpg"); // Replace "*.png" with the appropriate file extension
        
        if (cropBigImage) 
        {
            //D:\\OpenCV 2023\\Imgs\\MD122\\KPC3933_0\\annotations
            annotationFolderPath = newAnnotationFolderPath + "\\" + tumorFolder + "\\annotations";
            originalTiffFolderPath = newOriginalTiffFolderPath + "\\" + tumorFolder + "\\originals";
            filePaths = Directory.GetFiles(annotationFolderPath, "*.png"); // Replace "*.png" with the appropriate file extension
        }
        
        originalFilePaths = Directory.GetFiles(originalTiffFolderPath, "*.tif"); // Replace "*.png" with the appropriate file extension

        // Sort the file paths based on the numerical part of the file name
        Array.Sort(filePaths,
            (a, b) => int.Parse(Path.GetFileNameWithoutExtension(a))
                .CompareTo(int.Parse(Path.GetFileNameWithoutExtension(b))));
        /*else
        {
            filePaths = Array.FindAll(filePaths, file => Regex.IsMatch(Path.GetFileName(file), @"^Screenshot \(\d+\)\.png$"));
            Array.Sort(filePaths, CompareFilesBySuffixPNG);
        }*/
        
        filteredFiles = Array.FindAll(originalFilePaths, file => Regex.IsMatch(Path.GetFileName(file), @"^.*C1_\d+\.tif$"));

        // Sort the filtered files based on the numerical suffix
        Array.Sort(filteredFiles, CompareFilesBySuffix);

        
        //Debug.Log(originalFilePaths[0]);
        //Debug.Log(filePaths[0]);


        List<Mat> mats = new List<Mat>();
        List<Mat> croppedMats = new List<Mat>();
        foreach (string filePath in filePaths)
        {
            //Debug.Log(filePath);
            Mat img = Imgcodecs.imread(filePath);

            //Debug.Log(img.cols() + " "+ img.rows());
            if (cropBigImage)
            {
                //img = new Mat(img, new Rect(992, 759, 680, 698));
                img = new Mat(img, new Rect(750, 575, 1200, 1200));
            }

            /*if (cropBigImage)
            {
                Mat img2 = new Mat(img,new Rect(870,648,700,700));
                mats.Add(img2);
                Texture2D tex = new Texture2D(img2.width(), img2.height(), TextureFormat.RGB24, false);
                Utils.matToTexture2D(img2, tex);
                images.Add(tex);
                imagesCrop.Add(tex);

            }
            else
            {*/
                mats.Add(img);
                //croppedMats.Add(img);
                Texture2D tex = new Texture2D(img.width(), img.height(), TextureFormat.RGB24, false);
                Utils.matToTexture2D(img, tex);
                images.Add(tex);
                //imagesCrop.Add(tex);

            //}

        }

        currentImage = 0;
        
        List<Mat> blueShapes = new List<Mat>();
        blueShapes2 = new List<Mat>();
        blueShapes3 = new List<Mat>();
        int index = 0;
        foreach (Mat img in mats)
        {
            Mat hsv = new Mat();
            Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);
            Scalar lowerBlue = new Scalar(0, 100, 100);
            Scalar upperBlue = new Scalar(100, 255, 255);
            Mat blueMask = new Mat();
            Core.inRange(hsv, lowerBlue, upperBlue, blueMask);
            blueShapes.Add(blueMask);
            
            Texture2D tex = new Texture2D(blueMask.width(), blueMask.height(), TextureFormat.RGB24, false);
            Utils.matToTexture2D(blueMask, tex, false);
            
            //Utils.matToTexture2D(hsv, tex);
            blueShapesT.Add(tex);
            

            List<Rect> blueSquareRects = new List<Rect>();
            List<MatOfPoint> contours = new List<MatOfPoint>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(blueMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            Mat imgCopy = new Mat(img,new Rect(0,0,img.width(),img.height()));
            //img.copyTo(imgCopy);
            // Find the bounding rectangle for each contour and filter square shapes
            //Debug.Log(contours.Count);
            bool outsideSquare = false;
            foreach (MatOfPoint contour in contours)
            {
                Rect boundingRect = Imgproc.boundingRect(contour);
                float aspectRatio = (float) boundingRect.width / boundingRect.height;
                float area = boundingRect.width * boundingRect.height;

                // Adjust the aspectRatioThreshold and minAreaThreshold as needed
                float aspectRatioThreshold = 0.9f;
                float minAreaThreshold = 1000; // Adjust this value to filter small contours

                if (aspectRatio >= aspectRatioThreshold && aspectRatio <= (1 / aspectRatioThreshold) &&
                    area > minAreaThreshold && !outsideSquare)
                {
                    outsideSquare = true;
                    blueSquareRects.Add(boundingRect);
                    
                    // Draw the rectangle on the image
                    Scalar color = new Scalar(0, 0, 255); // Green color
                    int thickness = 2;
                    if (drawRect) Imgproc.rectangle(imgCopy, boundingRect.tl(), boundingRect.br(), color, thickness);
                    //Rect cropRect = boundingRect;
                    Rect cropRect = new Rect(boundingRect.x + 3, boundingRect.y + 3, boundingRect.width - 6,boundingRect.height - 6);
                    
                    Mat croppedImg = new Mat(img, cropRect);
                    croppedMats.Add(croppedImg);
                    Texture2D tex3 = new Texture2D(croppedImg.width(), croppedImg.height(), TextureFormat.RGB24, false);
                    Utils.matToTexture2D(croppedImg, tex3,false);
                    imagesCrop.Add(tex3);
                }
                else
                {
                    List<MatOfPoint> contourList = new List<MatOfPoint> { contour };
                    Scalar color = new Scalar(255, 0, 0); // Blue color
                    int thickness = 2;
                    Imgproc.drawContours(imgCopy, contourList, -1, color, thickness);
                }
            }
            Texture2D tex2 = new Texture2D(img.width(), img.height(), TextureFormat.RGB24, false);
            Utils.matToTexture2D(imgCopy, tex2,false);
            squareT.Add(tex2);

            Mat hsv2 = new Mat();
            Mat cropImg = croppedMats[^1];//croppedMats[index];//croppedMats[^1];
            index++;
            
            Imgproc.cvtColor(cropImg, hsv2, Imgproc.COLOR_BGR2HSV);
            Scalar lowerBlue2 = new Scalar(0, 100, 100);
            Scalar upperBlue2 = new Scalar(100, 255, 255);
            Mat blueMask2 = new Mat();
            Core.inRange(hsv2, lowerBlue2, upperBlue2, blueMask2);
            blueShapes2.Add(blueMask2);
            Texture2D texx = new Texture2D(blueMask2.width(), blueMask2.height(), TextureFormat.RGB24, false);
            Utils.matToTexture2D(blueMask2, texx, false);
            blueShapes2T.Add(texx);
            
            List<MatOfPoint> contours2 = new List<MatOfPoint>();
            Mat hierarchy2 = new Mat();
            Imgproc.findContours(blueMask2, contours2, hierarchy2, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            Mat imgCopy2 = new Mat(cropImg,new Rect(0,0,cropImg.width(),cropImg.height()));
            Mat imgCopy3 = new Mat(new Size(cropImg.width(),cropImg.height()),CvType.CV_8UC3);
            //img.copyTo(imgCopy);
            // Find the bounding rectangle for each contour and filter square shapes
            foreach (MatOfPoint contour in contours2)
            {
                List<MatOfPoint> contourList = new List<MatOfPoint> { contour };
                Scalar color = new Scalar(0, 0, 255); // Blue color
                Scalar whiteColor = new Scalar(255, 255, 255); // Blue color
                int thickness = -1;
                Imgproc.drawContours(imgCopy2, contourList, -1, color, thickness);
                Imgproc.drawContours(imgCopy3, contourList, -1, whiteColor, thickness);
            }
            blueShapes3.Add(imgCopy3);
        
            Texture2D texx2 = new Texture2D(cropImg.width(), cropImg.height(), TextureFormat.RGB24, false);
            Utils.matToTexture2D(imgCopy2, texx2,false);
            square2T.Add(texx2);
        }

        ready = true;
    }

    public void SavePreprocess()
    {
        int istart = 1;
        int iend = blueShapes3.Count;

        if (excludeNoAnnotation)
        {
            istart = startInclude;
            iend = endInclude;
        }

        int newResize = 384;
        if (resizeTo256) newResize = 256;

        for (int i = istart; i <=iend; i++)
        //foreach (Mat maskImage in blueShapes3)
        {
            Mat maskImage = blueShapes3[i - 1];
            if (maskImage.channels() != 1)
            {
                Imgproc.cvtColor(maskImage, maskImage, Imgproc.COLOR_BGR2GRAY);
            }

            // Resize the image to 384x384
            Size newSize = new Size(newResize, newResize);
            Mat resizedImage = new Mat(newSize, CvType.CV_8UC1);  // Ensure the resized image is also a single channel
            Imgproc.resize(maskImage, resizedImage, newSize);

            // Rotate 180 and flip vertically to correct orientation
            Core.rotate(resizedImage, resizedImage, Core.ROTATE_180);
            Core.flip(resizedImage, resizedImage, 1);  // 0 means flipping around x-axis (vertical flip)

            if (cropBigImage)
            {
                saveFolderPath = newSaveFolderPath + "\\" + tumorFolder + "\\All\\labels";
                saveImagesFolderPath = newSaveImagesFolderPath + "\\" + tumorFolder + "\\All\\images";
            }
            
            // Save the image to file as a PNG
            string filename = saveFolderPath + "\\" + prefix +"_"+ tumorConfig[activeTumor].day + "_"+ i + ".png";
            Imgcodecs.imwrite(filename, resizedImage);
            //i++;
        }

        for (int i = istart; i <=iend; i++)
        //foreach (var inputFilename in filteredFiles)
        {
            var inputFilename = filteredFiles[i - 1];
            Mat img = Imgcodecs.imread(inputFilename, Imgcodecs.IMREAD_UNCHANGED);
            Size newSize = new Size(newResize, newResize);
            Mat resizedImage = new Mat(newSize, CvType.CV_8UC3);  // Ensure the resized image is also a single channel
            Imgproc.resize(img, resizedImage, newSize);

            string filename = saveImagesFolderPath + "\\" + prefix+ "_"+tumorConfig[activeTumor].day +"_"+i + ".png";
            Imgcodecs.imwrite(filename, resizedImage);
            //i++;
        }
    }

    void DistributeImages(string sourceFolder, string destFolder1, string destFolder2, float percentage)
    {
        // List all PNG files in the source folder
        string[] files = Directory.GetFiles(sourceFolder, "*.png");

        // Shuffle the list of files
        var rand = new System.Random();
        files = files.OrderBy(x => rand.Next()).ToArray();

        // Calculate the split index based on the specified percentage
        int splitIndex = (int)(files.Length * (percentage / 100f));
        
        // Copy files to the first destination folder
        for (int i = 0; i < splitIndex; i++)
        {
            string destPath = Path.Combine(destFolder1, Path.GetFileName(files[i]));
            File.Copy(files[i], destPath, overwrite: true);
        }

        // Copy files to the second destination folder
        for (int i = splitIndex; i < files.Length; i++)
        {
            string destPath = Path.Combine(destFolder2, Path.GetFileName(files[i]));
            File.Copy(files[i], destPath, overwrite: true);
        }
    }
    
    public void GenerateTrainingSet()
    {
        Debug.Log("GENERATE ");

        if (cropBigImage)
        {
            trainFolderPath = newTrainFolderPath + "\\" + tumorFolder + "\\Train";
            testFolderPath = newTestFolderPath + "\\" + tumorFolder + "\\Test";
        }
        
        string imageTrainFolderPath = trainFolderPath + "\\images";
        string imageTestFolderPath = testFolderPath + "\\images";
        string labelTrainFolderPath = trainFolderPath + "\\labels";
        string labelTestFolderPath = testFolderPath + "\\labels";

        // List all PNG files in the source folder
        files = Directory.GetFiles(saveImagesFolderPath, "*.png");
        files2 = Directory.GetFiles(saveFolderPath, "*.png");

        if (cropBigImage)
        {
            newSaveImagesFolderPath = newSaveImagesFolderPath + "\\" + tumorFolder +"\\All\\images";
            newSaveFolderPath = newSaveFolderPath + "\\" + tumorFolder +"\\All\\labels";
            files = Directory.GetFiles(newSaveImagesFolderPath, "*.png");
            files2 = Directory.GetFiles(newSaveFolderPath, "*.png");
        }
        
        var filteredFiles1 = Array.FindAll(files, file => Regex.IsMatch(Path.GetFileName(file), @"^.*_\d+\.png$"));
        var filteredFiles2 = Array.FindAll(files2, file => Regex.IsMatch(Path.GetFileName(file), @"^.*_\d+\.png$"));

        // Sort the filtered files based on the numerical suffix
        //Array.Sort(filteredFiles1, CompareFilesBySuffixPNG2);
        //Array.Sort(filteredFiles2, CompareFilesBySuffixPNG2);

        
        // Shuffle the list of files
        var rand = new System.Random();
        //files = files.OrderBy(x => rand.Next()).ToArray();
        indices = Enumerable.Range(0, files.Length).OrderBy(x => rand.Next()).ToArray();

        // Rearrange both arrays based on the shuffled indices
        files = indices.Select(index => filteredFiles1[index]).ToArray();
        files2 = indices.Select(index => filteredFiles2[index]).ToArray();

        for (int i = 0; i < files.Length; i++)
        {
            string filename1 = Path.GetFileName(files[i]);
            string filename2 = Path.GetFileName(files2[i]);
            if (filename1 != filename2)
            {
                Debug.LogError("ERRRRRRRRORRRRR "+i+" "+files[i]+" "+files2[i]);
                return;
            }
        }
        
        // Calculate the split index based on the specified percentage
        int splitIndex = (int)(files.Length * (trainingSplit / 100f));
        //Debug.Log(splitIndex + " "+ files.Length);

        // Copy files to the first destination folder
        for (int i = 0; i < splitIndex; i++)
        {
            string destPath = Path.Combine(imageTrainFolderPath, Path.GetFileName(files[i]));
            File.Copy(files[i], destPath, overwrite: true);
            string destPath2 = Path.Combine(labelTrainFolderPath, Path.GetFileName(files2[i]));
            File.Copy(files2[i], destPath2, overwrite: true);
            //Debug.Log("Copy: "+destPath);
        }

        // Copy files to the second destination folder
        for (int i = splitIndex; i < files.Length; i++)
        {
            string destPath = Path.Combine(imageTestFolderPath, Path.GetFileName(files[i]));
            File.Copy(files[i], destPath, overwrite: true);
            string destPath2 = Path.Combine(labelTestFolderPath, Path.GetFileName(files2[i]));
            File.Copy(files2[i], destPath2, overwrite: true);
            //Debug.Log("Copy: "+destPath);
        }
        
        Debug.Log("OOOOOOOOOOOOOOOOOOOOOKKKKKKKKKKKKKKKKKK");

    }

    // Update is called once per frame
    void Update()
    {
        if (ready)
        {
            //uiMat.mainTexture = images[currentImage];
            if (viz == Visualize.IMAGE) rawImage.texture = images[currentImage];
            if (viz == Visualize.THRESHOLD) rawImage.texture = blueShapesT[currentImage];
            if (viz == Visualize.SQUARE_DET) rawImage.texture = squareT[currentImage];
            if (viz == Visualize.IMAGE_CROP) rawImage.texture = imagesCrop[currentImage];
            if (viz == Visualize.LABEL_DET) rawImage.texture = square2T[currentImage];
            if (viz == Visualize.ALL_CONTOURS) rawImage.texture = allContT[currentImage];
        }
    }
}
