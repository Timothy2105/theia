using UnityEngine;
using System;
using System.Net.Sockets;
using System.Threading;

public class TrackerGuy : MonoBehaviour
{
    public Transform dotOverlay;
    private MeshRenderer meshRenderer;
    private TcpClient client;
    private NetworkStream stream;
    private Thread readThread;
    private bool isRunning = true;
    private float xPos, yPos;
    private int state;
    private bool isConnected;
    private string serverIP = "127.0.0.1";
    private int serverPort = 2025;
    private bool shouldAttemptConnection = true;

    void Start()
    {
        try 
        {
            meshRenderer = dotOverlay.GetComponent<MeshRenderer>();
            StartConnectionAttempt();
            PositionInFrontOfCamera();
        }
        catch (Exception e)
        {
            Debug.LogError($"Start error: {e.Message}");
        }
    }

    void StartConnectionAttempt()
    {
        if (shouldAttemptConnection)
        {
            Thread connectionThread = new Thread(new ThreadStart(() => {
                try
                {
                    ConnectToServer();
                }
                catch (Exception e)
                {
                    Debug.LogError($"Connection thread error: {e.Message}");
                    isConnected = false;
                }
            }));
            connectionThread.IsBackground = true;
            connectionThread.Start();
        }
    }

    void ConnectToServer()
    {
        try
        {
            Debug.Log("Attempting to connect to server...");
            client = new TcpClient();
            var result = client.BeginConnect(serverIP, serverPort, null, null);
            var success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(1));
            
            if (!success)
            {
                Debug.LogWarning("Failed to connect to server, will retry...");
                shouldAttemptConnection = true;
                return;
            }

            client.EndConnect(result);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("Connected to server successfully!");

            readThread = new Thread(new ThreadStart(ReadData));
            readThread.IsBackground = true;
            readThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection error: {e.Message}");
            isConnected = false;
            shouldAttemptConnection = true;
        }
    }

    void ReadData()
    {
        byte[] buffer = new byte[12];
        while (isRunning && client != null && client.Connected)
        {
            try
            {
                if (stream.Read(buffer, 0, 12) == 12)
                {
                    if (BitConverter.IsLittleEndian)
                    {
                        Array.Reverse(buffer, 0, 4);
                        Array.Reverse(buffer, 4, 4);
                        Array.Reverse(buffer, 8, 4);
                    }
                    
                    xPos = BitConverter.ToSingle(buffer, 0);
                    yPos = BitConverter.ToSingle(buffer, 4);
                    float stateFloat = BitConverter.ToSingle(buffer, 8);
                    state = stateFloat > 0.5f ? 1 : 0;
                }
                Thread.Sleep(16);
            }
            catch (Exception e)
            {
                Debug.LogError($"Read error: {e.Message}");
                isConnected = false;
                break;
            }
        }
        shouldAttemptConnection = true;
    }

    void Update()
    {
        if (!isConnected && shouldAttemptConnection)
        {
            shouldAttemptConnection = false;
            StartConnectionAttempt();
            return;
        }

        if (!isConnected)
            return;

        try
        {
            Transform cameraTransform = GameObject.Find("OVRCameraRig")?.transform.Find("TrackingSpace/CenterEyeAnchor");
            if (cameraTransform != null)
            {
                Vector3 position = cameraTransform.position + cameraTransform.forward * 2f;
                position += new Vector3(0, yPos, xPos + 0.6f);
                transform.position = position;
            }
            if (meshRenderer != null)
            {
                Color currentColor = meshRenderer.material.color;
                currentColor.a = state == 1 ? 1f : 0f;  // 1 for visible, 0 for transparent
                meshRenderer.material.color = currentColor;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Update error: {e.Message}");
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        try
        {
            if (readThread != null)
            {
                readThread.Join(100); // Wait max 100ms
            }
            if (stream != null)
                stream.Close();
            if (client != null)
                client.Close();
        }
        catch (Exception e)
        {
            Debug.LogError($"Cleanup error: {e.Message}");
        }
    }
}