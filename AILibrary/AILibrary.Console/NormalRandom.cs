namespace AILibrary;

public class NormalRandom
{
    private Random _rand;
    private bool _hasSpare;
    private double _spare;

    public NormalRandom()
    {
        _rand = new Random();
        _hasSpare = false;
    }

    public NormalRandom(int seed)
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
}
