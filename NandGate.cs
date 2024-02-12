namespace Gates;
public class NandGate(float[,] testing)
{
    private const int Count = 4;

    public float Cost(float w1, float w2, float b) {
        var res = 0.0f;
        for (var i = 0; i < Count; ++i) {
            var x1 = testing[i, 0];
            var x2 = testing[i, 1];
            var y = Sigmoid_f(x1 * w1 + x2 * w2 + b);
            var d = y - testing[i, 2];
            res += d * d;
        }
        res /= Count;
        return res;
    }
    private static float Sigmoid_f(float x) {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }
}
