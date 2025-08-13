namespace AILibrary;

public class RandomNeuron
{
    private Random _rand;
    private bool _hasSpare;
    private float _spare;

    public RandomNeuron()
    {
        _rand = new Random();
        _hasSpare = false;
    }

    public RandomNeuron(int seed)
    {
        _rand = new Random(seed);
        _hasSpare = false;
    }

    public float NextStandardNormal()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }

        float u, v, s;
        do
        {
            u = (float) _rand.NextDouble() * 2.0F - 1.0F; // uniform(-1, 1)
            v = (float) _rand.NextDouble() * 2.0F - 1.0F;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0);

        float multiplier = (float) Math.Sqrt(-2.0 * Math.Log(s) / s);
        _spare = v * multiplier;
        _hasSpare = true;

        return u * multiplier; // standard normal (mean=0, std=1)
    }

    public static float[] CreateRandomNeurons(int x, bool useSeed, int seed = 0)
    {
        RandomNeuron rng = useSeed ? new RandomNeuron(seed) : new RandomNeuron();

        float[] output = new float[x];

        for (int i = 0; i < x; i++)
        {
            output[i] = rng.NextStandardNormal();
        }

        return output;
    }

    public static float[][] CreateRandomNeurons(int x, int y, bool useSeed, int seed = 0)
    {
        RandomNeuron rng = useSeed ? new RandomNeuron(seed) : new RandomNeuron();

        float[][] output = new float[x][];

        for (int i = 0; i < x; i++)
        {
            output[i] = new float[y];

            for (int j = 0; j < y; j++)
            {
                output[i][j] = rng.NextStandardNormal();
            }
        }

        return output;
    }
}
