namespace AILibrary;

public static class SystemModel
{
    public static char[] Alphabet = new char[27] // itos
    {
        '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    };

    public static Dictionary<char, int> AlphabetNumbers = GetAlphabetNumbers();

    static Dictionary<char, int> GetAlphabetNumbers()
    {
        Dictionary<char, int> output = new Dictionary<char, int>();

        for (int i = 0; i < Alphabet.Length; i++)
        {
            output.Add(Alphabet[i], i);
        }

        return output;
    }
}
