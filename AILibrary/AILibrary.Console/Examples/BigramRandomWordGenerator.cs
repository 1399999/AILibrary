using System.Text.Json;

namespace AILibrary.Examples;

public class BigramRandomWordGenerator
{
    private Dictionary<char, Dictionary<char, int>> bigramCounts = new();
    private Random random = new();

    public BigramRandomWordGenerator() { }

    public BigramRandomWordGenerator(string path)
    {
        bigramCounts = JsonSerializer.Deserialize<Dictionary<char, Dictionary<char, int>>>(File.ReadAllText(path));
    }

    public void Train(IEnumerable<string> words, string? saveModelPath = null)
    {
        var wordList = words.ToList();

        for (int i = 0; i < words.Count(); i++)
        {
            string w = "<" + wordList[i].ToLower() + " >"; // < = start, > = end

            for (int j = 0; j < w.Length - 1; j++)
            {
                char first = w[j];
                char second = w[j + 1];

                if (!bigramCounts.ContainsKey(first))
                    bigramCounts[first] = new Dictionary<char, int>();

                if (!bigramCounts[first].ContainsKey(second))
                    bigramCounts[first][second] = 0;

                bigramCounts[first][second]++;
            }
        }

        if (saveModelPath != null && Directory.Exists(Path.GetPathRoot(saveModelPath)))
        {
            JsonEncode(saveModelPath, bigramCounts);
        }
    }

    public string GenerateWord()
    {
        char current = '<'; // Start symbol
        string result = "";

        while (true)
        {
            if (!bigramCounts.ContainsKey(current))
                break;

            var nextChar = WeightedRandomChoice(bigramCounts[current]);
            if (nextChar == '>')
                break;

            result += nextChar;
            current = nextChar;
        }

        return result;
    }

    private char WeightedRandomChoice(Dictionary<char, int> choices)
    {
        int total = choices.Values.Sum();
        int r = random.Next(total);
        int sum = 0;

        foreach (var kv in choices)
        {
            sum += kv.Value;
            if (r < sum)
                return kv.Key;
        }
        return choices.Keys.First(); // Fallback
    }

    public static void JsonEncode(string path, Dictionary<char, Dictionary<char, int>> input) => File.WriteAllText(path, JsonSerializer.Serialize(input));

    public static void RunExample()
    {
        var words = new List<string>
        {
            "cat", "car", "cap", "dog", "dot", "door", "xylophone", "anomaly"
        };

        var generator = new BigramRandomWordGenerator();
        generator.Train(words);

        Console.WriteLine("Generated words:");

        for (int i = 0; i < 50; i++)
        {
            Console.WriteLine(generator.GenerateWord());
        }
    }
}
