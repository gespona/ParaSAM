using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VoxelPlay;

public class TestVoxel : MonoBehaviour
{
    public VoxelPlayEnvironment env;

    public VoxelDefinition vd;
    // Start is called before the first frame update
    void Start()
    {
        
        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                for (int k = 0; k < 200; k++)
                {
                    env.VoxelPlace(new Vector3(i,j,k), vd);
                }
            }
        }
        

    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
