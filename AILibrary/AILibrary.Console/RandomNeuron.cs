namespace AILibrary;

public class RandomNeuron
{
    private Random _rand;
    private bool _hasSpare;
    private double _spare;

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

    public double NextStandardNormal()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }

        double u, v, s;
        do
        {
            u = _rand.NextDouble() * 2.0 - 1.0; // uniform(-1, 1)
            v = _rand.NextDouble() * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0);

        double multiplier = Math.Sqrt(-2.0 * Math.Log(s) / s);
        _spare = v * multiplier;
        _hasSpare = true;

        return u * multiplier; // standard normal (mean=0, std=1)
    }

    public static double[] CreateRandomNeurons(int x, bool useSeed, int seed = 0)
    {
        RandomNeuron rng = useSeed ? new RandomNeuron(seed) : new RandomNeuron();

        double[] output = new double[x];

        for (int i = 0; i < x; i++)
        {
            output[i] = rng.NextStandardNormal();
        }

        return output;
    }

    public static double[][] CreateRandomNeurons(int x, int y, bool useSeed, int seed = 0)
    {
        RandomNeuron rng = useSeed ? new RandomNeuron(seed) : new RandomNeuron();

        double[][] output = new double[x][];

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < x; j++)
            {
                output[i][j] = rng.NextStandardNormal();
            }
        }

        return output;
    }
}
