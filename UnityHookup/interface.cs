using UnityEngine;
using System.IO;

public class DataReader : MonoBehaviour
{
    public string filePath = "live-demo\live-predictions.txt";
    public Transform dotOverlay;
    private float lastModified = 0f;

    void Update()
    {
        if (File.GetLastWriteTime(filePath).ToFileTime() > lastModified)
        {
            string[] data = File.ReadAllText(filePath).Split(',');
            int state = int.Parse(data[0]);
            float x = float.Parse(data[1]);
            float y = float.Parse(data[2]);

            dotOverlay.position = new Vector3(x, y, 0);
            dotOverlay.gameObject.SetActive(state == 1);

            lastModified = File.GetLastWriteTime(filePath).ToFileTime();
        }
    }
}
