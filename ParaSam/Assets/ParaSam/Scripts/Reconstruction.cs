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
using TMPro;
using VoxelPlay;

[System.Serializable]

public class Reconstruction : MonoBehaviour
{
    public ImageLoader imageLoader;

    public bool ready;
    [Header("Load Inference")]
    public string originalTiffFolderPath;
    public string originalMaskFolderPath;
    public string inferenceFolderPath;
    public string[] filePaths;
    public string[] originalMaskPaths;
    public string[] inferencePaths;
    public string[] filteredFilePaths;
    public string[] filteredOriginalMaskPaths;
    public string[] filteredInferencePaths;

    
    public List<Mat> images;
    public List<Mat> masks;
    public List<Mat> masksGT;
    public List<Mat> matMaskViz;

    public List<Texture2D> maskViz;

    [Header("Reconstruction")] 
    public List<Vector2> mask1;
    public List<Vector2> mask2;

    public static float _sliceDistance = 200; // The distance between the masks along the y-axis

    // Create a 3D voxel grid, with a depth based on the sliceDistance
    // Assuming the voxel grid size is known and corresponds to the real-world size it represents
    public bool[,,] voxelGrid = new bool[256, 256, Mathf.CeilToInt(_sliceDistance) ];

    public int totalVolume;
    public GameObject voxelPrefab;
    public GameObject voxelReconstructionParent;
    
    public VoxelPlayEnvironment env;
    public VoxelDefinition vd;

    public GameObject canvas3D;
    public TextMeshProUGUI volumeText;

    public bool playAnimation;
    public float animationFrequency = 0.25f;
    public float timeElapsed;
    
    public ParticleSystem voxelParticleSystem; // Assign this in the Inspector
    private ParticleSystem.Particle[] particles;
    
    
    
    void Start()
    {
        ready = false;
        playAnimation = false;
        
        var main = voxelParticleSystem.main;
        main.loop = false;
        main.startLifetime = Mathf.Infinity;
        main.startSpeed = 0;

        var emission = voxelParticleSystem.emission;
        emission.rateOverTime = 0;
        emission.SetBursts(new ParticleSystem.Burst[] { new ParticleSystem.Burst(0.0f, 655360) });

        var shape = voxelParticleSystem.shape;
        shape.shapeType = ParticleSystemShapeType.Box; // Example: Box shape

        voxelParticleSystem.Play();
    }

    int CompareFilesBySuffix(string a, string b)
    {
        try
        {
            int suffixA = int.Parse(Regex.Match(Path.GetFileName(a), @"_(\d+)\.png$").Groups[1].Value);
            int suffixB = int.Parse(Regex.Match(Path.GetFileName(b), @"_(\d+)\.png$").Groups[1].Value);
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

    public void LoadInference()
    {
        ready = false;
        
        images = new List<Mat>();
        masks = new List<Mat>();
        masksGT = new List<Mat>();
        matMaskViz = new List<Mat>();
        maskViz = new List<Texture2D>();
        
        originalTiffFolderPath = imageLoader.newSaveFolderPath + "\\" + imageLoader.tumorFolder + "\\All\\images";
        filePaths = Directory.GetFiles(originalTiffFolderPath, "*.png"); // Replace "*.png" with the appropriate file extension

        originalMaskFolderPath = imageLoader.newSaveFolderPath + "\\" + imageLoader.tumorFolder + "\\All\\labels";
        originalMaskPaths = Directory.GetFiles(originalMaskFolderPath, "*.png"); // Replace "*.png" with the appropriate file extension
        
        inferenceFolderPath = imageLoader.newSaveFolderPath + "\\" + imageLoader.tumorFolder + "\\inference";
        inferencePaths = Directory.GetFiles(inferenceFolderPath, "*.png"); // Replace "*.png" with the appropriate file extension

        filteredFilePaths = Array.FindAll(filePaths, file => Regex.IsMatch(Path.GetFileName(file), @"^.*_\d+\.png$"));
        filteredOriginalMaskPaths = Array.FindAll(originalMaskPaths, file => Regex.IsMatch(Path.GetFileName(file), @"^.*_\d+\.png$"));
        filteredInferencePaths = Array.FindAll(inferencePaths, file => Regex.IsMatch(Path.GetFileName(file), @"^.*_\d+\.png$"));

        Array.Sort(filteredFilePaths, CompareFilesBySuffix);
        Array.Sort(filteredOriginalMaskPaths, CompareFilesBySuffix);
        Array.Sort(filteredInferencePaths, CompareFilesBySuffix);

        foreach (string filePath in filteredFilePaths)
        {
            //Debug.Log(filePath);
            Mat img = Imgcodecs.imread(filePath);
            images.Add(img);
        }

        foreach (string filePath in filteredOriginalMaskPaths)
        {
            //Debug.Log(filePath);
            Mat img = Imgcodecs.imread(filePath);
            masksGT.Add(img);
        }

        foreach (string filePath in filteredInferencePaths)
        {
            //Debug.Log(filePath);
            Mat img = Imgcodecs.imread(filePath);
            masks.Add(img);
        }

        // prepare viz

        for (int i = 0; i < filePaths.Length; i++)
        {
            // Overlay the mask onto the RGB image with some transparency
            Mat maskOverlay = new Mat();
            Core.addWeighted(images[i], 1.0, masks[i], 1.0, 0.0, maskOverlay);

            // Assuming masksGT[i] is the current mask you want to process
            Mat grayMat = new Mat();
            Imgproc.cvtColor(masksGT[i], grayMat, Imgproc.COLOR_RGB2GRAY); // Convert to grayscale if it's a color image

            Mat binaryMat = new Mat();
            Imgproc.threshold(grayMat, binaryMat, 127, 255, Imgproc.THRESH_BINARY); // Apply threshold

            // Find GT mask contours for overlay
            List<MatOfPoint> gtContours = new List<MatOfPoint>();
            Imgproc.findContours(binaryMat, gtContours, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);

            // Draw contours onto the RGB image
            foreach (MatOfPoint gtContour in gtContours)
            {
                Imgproc.drawContours(maskOverlay, gtContours, -1, new Scalar(0, 0, 255), 1);
            }

            matMaskViz.Add(maskOverlay);

            Texture2D resultTexture =
                new Texture2D(maskOverlay.cols(), maskOverlay.rows(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(maskOverlay, resultTexture);

            maskViz.Add(resultTexture);

        }

        ready = true;

    }

    
    List<Vector2> ExtractMaskPoints(Mat mask)
    {
        List<Vector2> points = new List<Vector2>();
        for (int y = 0; y < mask.rows(); y++)
        {
            for (int x = 0; x < mask.cols(); x++)
            {
                // Check if the pixel is part of the mask (assuming white means part of the mask)
                // You might need to adjust this condition based on your specific mask format
                if (mask.get(y, x)[0] == 255) // Assuming mask is a grayscale image
                {
                    points.Add(new Vector2(x, y));
                }
            }
        }
        return points;
    }
    
    // Method to find the closest point on the target mask for a given source point
    Vector2 FindClosestPoint(Vector2 sourcePoint, List<Vector2> targetMask)
    {
        Vector2 closestPoint = targetMask[0];
        float closestDistanceSqr = (closestPoint - sourcePoint).sqrMagnitude;
    
        foreach (Vector2 targetPoint in targetMask)
        {
            float distanceSqr = (targetPoint - sourcePoint).sqrMagnitude;
            if (distanceSqr < closestDistanceSqr)
            {
                closestDistanceSqr = distanceSqr;
                closestPoint = targetPoint;
            }
        }
    
        return closestPoint;
    }
    
    // Method to fill the voxel corresponding to the point
    void FillVoxel(Vector3 point, bool[,,] grid)
    {
        int voxelX = Mathf.FloorToInt(point.x);
        int voxelY = Mathf.FloorToInt(point.y);
        int voxelZ = Mathf.FloorToInt(point.z);

        if (voxelX >= 0 && voxelX < grid.GetLength(0) &&
            voxelY >= 0 && voxelY < grid.GetLength(1) &&
            voxelZ >= 0 && voxelZ < grid.GetLength(2))
        {
            grid[voxelX, voxelY, voxelZ] = true;
        }
    }
    
    void InterpolateAndFillVoxels(List<Vector2> sourceMask, List<Vector2> targetMask, float distance)
    {
        int i = 0;
        foreach (Vector2 sourcePoint in sourceMask)
        {
            //Debug.Log("POINT "+i+ " OUT OF "+sourceMask.Count);
            Vector2 closestTargetPoint = FindClosestPoint(sourcePoint, targetMask);
            Vector3 startPoint = new Vector3(sourcePoint.x, 0, sourcePoint.y);
            Vector3 endPoint = new Vector3(closestTargetPoint.x, distance, closestTargetPoint.y);

            float step = 0.5f;//1.0f / Mathf.Abs(distance); // Use the absolute value of distance for the step
            for (float t = 0; t <= 1; t += step)
            {
                Vector3 interpolatedPoint = Vector3.Lerp(startPoint, endPoint, t);
                FillVoxel(interpolatedPoint, voxelGrid);
            }

            i++;
        }
    }
    
    // Interpolation and voxel filling between two masks
    /*void InterpolateAndFillVoxels(List<Vector2> sourceMask, List<Vector2> targetMask, float distance)
    {
        int i = 0;
        foreach (Vector2 sourcePoint in sourceMask)
        {
            Debug.Log("POINT "+i+ " OUT OF "+sourceMask.Count);
            Vector2 closestTargetPoint = FindClosestPoint(sourcePoint, targetMask);
            Vector3 startPoint = new Vector3(sourcePoint.x, 0, sourcePoint.y);
            Vector3 endPoint = new Vector3(closestTargetPoint.x, distance, closestTargetPoint.y);

            Debug.Log("FILL VOX");
            for (float t = 0; t <= 1; t += 1.0f / distance)
            {
                Vector3 interpolatedPoint = Vector3.Lerp(startPoint, endPoint, t);
                FillVoxel(interpolatedPoint, voxelGrid);
            }

            i++;
        }
    }*/
    
    // Calculate the volume of the filled voxels
    int CalculateVolume(bool[,,] grid)
    {
        int count = 0;
        foreach (bool voxel in grid)
        {
            if (voxel) count++;
        }
        return count;
    }

    int CalculateVolume(bool[,,] grid, int layer)
    {
        int count = 0;
        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                for (int k = 0; k < layer; k++)
                {
                    if (grid[i,j,k]) count++;
                }
            }
        }
        return count;
    }


    public void FillVoxels(List<Vector2> sourceMask, int layer)
    {
        foreach (Vector2 sourcePoint in sourceMask)
        {
            env.VoxelPlace(new Vector3(sourcePoint.x,sourcePoint.y,layer), vd);
            voxelGrid[(int)sourcePoint.x,(int)sourcePoint.y,layer] = true;
        }
    }
    
    public void RunReconstruction2()
    {
        for (int i = 0; i < masks.Count; i++)
        {
            Debug.Log("LAYER " + i);
            Mat maskMat1 = masks[i];
            Core.flip(maskMat1, maskMat1, 0); // 0 means flipping around x-axis (vertical flip)
            mask1 = ExtractMaskPoints(maskMat1);

            FillVoxels(mask1, i);

        }

        /*totalVolume = CalculateVolume(voxelGrid);
        volumeText.SetText("Tumor Volume: " + totalVolume * 0.000169172f + " mm^3");*/

        imageLoader.currentImage = 0;
        playAnimation = true;

    }
    
    public void RunReconstruction()
    {
        for (int i = 0; i < masks.Count - 1; i++)
        {
            Debug.Log("LAYER "+i);
            Mat maskMat1 = masks[i];
            Mat maskMat2 = masks[i+1];

            mask1 = ExtractMaskPoints(maskMat1);
            mask2 = ExtractMaskPoints(maskMat2);

            //Debug.Log("MASK1");
            // First pass: mask1 to mask2
            InterpolateAndFillVoxels(mask1, mask2, _sliceDistance);

            //Debug.Log("MASK2");

            // Second pass: mask2 to mask1 (reverse direction)
            InterpolateAndFillVoxels(mask2, mask1, -_sliceDistance);
            
            VisualizeVoxelsPrefab((int) (i * _sliceDistance));
            
            totalVolume += CalculateVolume(voxelGrid);
            
            CleanVoxelGrid();
            mask1.Clear();
            mask2.Clear();
        }

        
        //VisualizeVoxels(voxelGrid);
        
        //VisualizeVoxelsPrefab();
    }
    
    private int CountVoxels(bool[,,] grid)
    {
        int count = 0;
        foreach (bool voxel in grid)
        {
            if (voxel) count++;
        }
        return count;
    }
    
    public void VisualizeVoxels(bool[,,] voxelGrid)
    {
        int numVoxels = CountVoxels(voxelGrid);
        //Debug.Log("numVox: "+numVoxels);
        particles = new ParticleSystem.Particle[numVoxels];

        int particleIndex = 0;
        for (int x = 0; x < voxelGrid.GetLength(0); x++)
        {
            for (int y = 0; y < voxelGrid.GetLength(1); y++)
            {
                for (int z = 0; z < voxelGrid.GetLength(2); z++)
                {
                    if (voxelGrid[x, y, z])
                    {
                        particles[particleIndex].position = new Vector3(x, y, z);
                        particles[particleIndex].startColor = new Color(1.0f, 1.0f, 1.0f);
                        particles[particleIndex].startSize = 1.0f; // Adjust size as needed
                        particleIndex++;
                    }
                }
            }
        }

        Debug.Log(particleIndex+" "+particles.Length);
        voxelParticleSystem.SetParticles(particles, particles.Length);
    }

    private void CleanVoxelGrid()
    {
        for (int x = 0; x < voxelGrid.GetLength(0); x++)
        {
            for (int y = 0; y < voxelGrid.GetLength(1); y++)
            {
                for (int z = 0; z < voxelGrid.GetLength(2); z++)
                {
                    voxelGrid[x, y, z] = false;
                }
            }

        }
    }

    private void VisualizeVoxelsPrefab(int offset)
    {
        GameObject go = new GameObject();
        go.name = "layer_" + offset;
        go.transform.parent = voxelReconstructionParent.transform;
        
        for (int x = 0; x < voxelGrid.GetLength(0); x++)
        {
            for (int y = 0; y < voxelGrid.GetLength(1); y++)
            {
                for (int z = 0; z < voxelGrid.GetLength(2); z++)
                {
                    if (voxelGrid[x, y, z])
                    {
                        GameObject voxel = Instantiate(voxelPrefab, new Vector3(x, y-offset, z), Quaternion.identity);
                        voxel.transform.parent = go.transform;
                    }
                }
            }
        }
    }
    
    void Update()
    {
        if (ready && !playAnimation)
        {
            imageLoader.rawImage.texture = maskViz[imageLoader.currentImage];
            canvas3D.transform.position = new Vector3(128.0f, 128.0f, imageLoader.currentImage+ 0.9f);
        }

        if (ready && playAnimation)
        {
            timeElapsed += Time.deltaTime;
            if (timeElapsed > animationFrequency)
            {
                imageLoader.currentImage++;
                if (imageLoader.currentImage < filePaths.Length)
                {
                    imageLoader.rawImage.texture = maskViz[imageLoader.currentImage];
                    canvas3D.transform.position = new Vector3(128.0f, 128.0f, imageLoader.currentImage+ 0.9f);
                    totalVolume = CalculateVolume(voxelGrid,imageLoader.currentImage);
                    volumeText.SetText("Tumor Volume: " + totalVolume * 0.000169172f + " mm^3");
                }
                else
                {
                    playAnimation = false;
                    imageLoader.currentImage--;
                    /*imageLoader.currentImage = 0;
                    imageLoader.rawImage.texture = maskViz[imageLoader.currentImage];
                    canvas3D.transform.position = new Vector3(128.0f, 128.0f, imageLoader.currentImage+ 0.9f);*/
                }

                timeElapsed = 0.0f;
            }
        }
    }
}