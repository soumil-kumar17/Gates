namespace Gates;
public class AndGate {
    private readonly float[,] _testing = new float[4, 3];
    private const int Count = 4;
    public AndGate(float[,] testing) {
        _testing = testing;
    }
    public float Cost(float w1, float w2, float b) {
        float res = 0.0f;
        for (int i = 0; i < Count; ++i) {
            float x1 = _testing[i, 0];
            float x2 = _testing[i, 1];
            float y = Sigmoid_f(x1*w1 + x2*w2 + b);
            float d = y - _testing[i, 2];
            res += d * d;
        }
        res /= Count;
        return res;
    }

    private static float Sigmoid_f(float x) {
        return 1.0f/(1.0f + (float)Math.Exp(-x));
    }
}