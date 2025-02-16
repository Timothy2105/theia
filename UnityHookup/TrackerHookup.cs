using UnityEngine;
using System.IO;
using System.Text;

public class DataReader : MonoBehaviour
{
    [SerializeField] 
    private string filePath;
    public Transform dotOverlay;
    private string lastContent = "";

    void Start()
    {
        // Set the path locally for now because I am bad at coding.
        filePath = "C:/Users/Jared/Desktop/theia/live-demo/live-predictions.txt";
        Debug.Log($"Starting DataReader with path: {filePath}");
        Debug.Log($"File exists: {File.Exists(filePath)}");

        Directory.CreateDirectory(Path.GetDirectoryName(filePath));

        if (!File.Exists(filePath))
        {
            Debug.Log("Creating new file");
            File.WriteAllText(filePath, "0 0.0 0.0", Encoding.UTF8);
        }

        
        try
        {
            string contents = File.ReadAllText(filePath);
            Debug.Log($"Initial file contents: {contents}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to read file: {e.Message}");
        }
    }

    void Update()
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Debug.LogWarning("File does not exist!");
                return;
            }
            // Read with explicit UTF8 encoding
            string currentContent = File.ReadAllText(filePath, Encoding.UTF8);
            // Only process if content changed
            if (currentContent == lastContent) return;
            Debug.Log($"Reading content: {currentContent}");  // Debug line
                                                              // Clean the input string
            currentContent = currentContent.Trim();
            string[] data = currentContent.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (data.Length < 3)
            {
                Debug.LogWarning($"Not enough values in file. Found {data.Length} values, need 3");
                return;
            }
            // Try parse each value separately for better error reporting
            if (!int.TryParse(data[0], out int state))
            {
                Debug.LogWarning($"Failed to parse state value: {data[0]}");
                return;
            }
            if (!float.TryParse(data[1], out float x))
            {
                Debug.LogWarning($"Failed to parse x value: {data[1]}");
                return;
            }
            if (!float.TryParse(data[2], out float y))
            {
                Debug.LogWarning($"Failed to parse y value: {data[2]}");
                return;
            }
            // All parsing successful, update position
            dotOverlay.position = new Vector3(x, y, 0);
            dotOverlay.gameObject.SetActive(state == 1);
            lastContent = currentContent;
            Debug.Log($"Updated position to: {x}, {y} with state {state}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error in DataReader: {e.Message}\nStack trace: {e.StackTrace}");
        }
    }
}