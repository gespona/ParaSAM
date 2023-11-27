using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;

public class UIManager : MonoBehaviour
{
    public ImageLoader imageLoader;


    public bool useReconstruction;
    public Reconstruction reconstruction;
    
    public TextMeshProUGUI text;
    
    // Start is called before the first frame update
    void Start()
    {
        text.text = imageLoader.tumorConfig[imageLoader.activeTumor].prefix+"_"+imageLoader.tumorConfig[imageLoader.activeTumor].day;
    }

    public void NextImage()
    {
        imageLoader.currentImage++;
        if (useReconstruction)
        {
            if (imageLoader.currentImage >= reconstruction.filePaths.Length) imageLoader.currentImage = 0;
        }
        else
        {
            if (imageLoader.currentImage >= imageLoader.filePaths.Length) imageLoader.currentImage = 0;
        }
    }

    public void PrevImage()
    {
        imageLoader.currentImage--;
        if (useReconstruction)
        {
            if (imageLoader.currentImage < 0) imageLoader.currentImage = reconstruction.filePaths.Length - 1;
        }
        else
        {
            if (imageLoader.currentImage < 0) imageLoader.currentImage = imageLoader.filePaths.Length - 1;
        }
    }

    public void NextTumor()
    {
        imageLoader.activeTumor++;
        text.text = imageLoader.tumorConfig[imageLoader.activeTumor].prefix+"_"+imageLoader.tumorConfig[imageLoader.activeTumor].day;
    }

    public void PrevTumor()
    {
        imageLoader.activeTumor--;
        text.text = imageLoader.tumorConfig[imageLoader.activeTumor].prefix+"_"+imageLoader.tumorConfig[imageLoader.activeTumor].day;
    }

    public void Preprocess()
    {
        imageLoader.Preprocess();
    }

    
    public void SavePreprocess()
    {
        imageLoader.SavePreprocess();
    }

    public void GenerateTrainingSet()
    {
        imageLoader.GenerateTrainingSet();
    }

    public void LoadInference()
    {
        reconstruction.LoadInference();
    }

    public void RunReconstruction()
    {
        reconstruction.RunReconstruction2();
    }

    public void ShowHideReconstruction()
    {
        reconstruction.env.worldRoot.gameObject.SetActive(!reconstruction.env.worldRoot.gameObject.activeSelf);
    }
    
    public void ShowHideImages()
    {
        reconstruction.canvas3D.SetActive(!reconstruction.canvas3D.activeSelf);
    }
    
    // Update is called once per frame
    void Update()
    {
        
    }
}
