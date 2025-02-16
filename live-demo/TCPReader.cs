using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.IO;

public class TCPReader
{
    private const string FilePath = "live-predictions.txt";
    private const string ServerIP = "127.0.0.1";  // Use localhost for testing
    private const int ServerPort = 2025;
    
    public void StartClient()
    {
        try
        {
            Console.WriteLine("Connecting to server...");
            TcpClient client = new TcpClient();
            client.Connect(ServerIP, ServerPort);
            Console.WriteLine("Connected!");
           
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[12];
            while (true)
            {
                if (stream.Read(buffer, 0, 12) == 12)
                {
                    // Convert from network byte order
                    if (BitConverter.IsLittleEndian)
                    {
                        Array.Reverse(buffer, 0, 4);
                        Array.Reverse(buffer, 4, 4);
                        Array.Reverse(buffer, 8, 4);
                    }
                   
                    float x = BitConverter.ToSingle(buffer, 0);
                    float y = BitConverter.ToSingle(buffer, 4);
                    float state = BitConverter.ToSingle(buffer, 8);
                   
                    // Write to file in the format Unity expects
                    string outputData = string.Format("{0} {1:F1} {2:F1}",
                        (state > 0.5f ? 1 : 0), x, y);
                    File.WriteAllText(FilePath, outputData);
                   
                    Console.WriteLine("Updated: " + outputData);
                }
                Thread.Sleep(16);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: " + e.Message);
            Console.WriteLine("Retrying in 5 seconds...");
            Thread.Sleep(5000);
        }
    }

    public static void Main()
    {
        new TCPReader().StartClient();
    }
}