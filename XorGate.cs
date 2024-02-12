namespace Gates;
public class XorGate {
    public float OrW1 { get; private set; }
    public float OrW2 { get; private set; }
    public float OrB { get; private set; }
    public float NandW1 { get; private set; }
    private float NandW2 { get; set; }
    public float NandB { get; private set; }
    public float AndW1 { get; private set; }
    private float AndW2 { get; set; }
    public float AndB { get; private set; }

    private readonly float[,] _testing = new float[4, 3];
    private const int Count = 4;
    public XorGate(float[,] testing) {
        _testing = testing;
    }
    private XorGate() { }

    public static float Forward(XorGate model, float x1, float x2) {
        var a = Sigmoid_f(model.OrW1*x1 + model.OrW2*x2 + model.OrB);
        var b = Sigmoid_f(model.NandW1*x1 + model.NandW1 * x2 + model.NandB);
        return Sigmoid_f(model.AndW1*a + model.AndW2*b + model.AndB);
    }

    public static void Init_Rand_Xor(XorGate xor) {
        Random random = new();
        xor.OrW1 = (float)random.NextDouble();
        xor.OrW2 = (float)random.NextDouble();
        xor.OrB = (float)random.NextDouble();
        xor.NandW1 = (float)random.NextDouble();
        xor.NandW2 = (float)random.NextDouble();
        xor.NandB = (float)random.NextDouble();
        xor.AndW1 = (float)random.NextDouble();
        xor.AndW2 = (float)random.NextDouble();
        xor.AndB = (float)random.NextDouble();
    }

    public static void Print_Xor(XorGate xor) {
        Console.WriteLine(xor.OrW1);
        Console.WriteLine(xor.OrW2);
        Console.WriteLine(xor.OrB);
        Console.WriteLine(xor.NandW1);
        Console.WriteLine(xor.NandW2);
        Console.WriteLine(xor.NandB);
        Console.WriteLine(xor.AndW1);
        Console.WriteLine(xor.AndW2);
        Console.WriteLine(xor.AndB);
    }

    public XorGate Finite_diff(XorGate m, float eps) {
        XorGate g = new();
        var c = Cost(m);

        var saved = m.OrW1;
        m.OrW1 += eps;
        g.OrW1 = (Cost(m) - c) / eps;
        m.OrW1 = saved;

        saved = m.OrW2;
        m.OrW2 += eps;
        g.OrW2 = (Cost(m) - c) / eps;
        m.OrW2 = saved;

        saved = m.OrB;
        m.OrB += eps;
        g.OrB = (Cost(m) - c) / eps;
        m.OrB = saved;

        saved = m.NandW1;
        m.NandW1 += eps;
        g.NandW1 = (Cost(m) - c) / eps;
        m.NandW1 = saved;

        saved = m.NandW2;
        m.NandW2 += eps;
        g.NandW2 = (Cost(m) - c) / eps;
        m.NandW2 = saved;

        saved = m.NandB;
        m.NandB += eps;
        g.NandB = (Cost(m) - c) / eps;
        m.NandB = saved;

        saved = m.AndW1;
        m.AndW1 += eps;
        g.AndW1 = (Cost(m) - c) / eps;
        m.AndW1 = saved;

        saved = m.AndW2;
        m.AndW2 += eps;
        g.AndW2 = (Cost(m) - c) / eps;
        m.AndW2 = saved;

        saved = m.AndB;
        m.AndB += eps;
        g.AndB = (Cost(m) - c) / eps;
        m.AndB = saved;

        return g;
    }

    public static XorGate TrainModel(XorGate m, XorGate g, float rate) {
        m.OrW1 -= rate * g.OrW1;
        m.OrW2 -= rate * g.OrW2;
        m.OrB -= rate * g.OrB;
        m.NandW1 -= rate * g.NandW1;
        m.NandW2 -= rate * g.NandW2;
        m.NandB -= rate * g.NandB;
        m.AndW1 -= rate * g.AndW1;
        m.AndW2 -= rate * g.AndW2;
        m.AndB -= rate * g.AndB;
        return m;
    }

    public float Cost(XorGate model) {
        var res = 0.0f;
        for (var i = 0; i < Count; ++i) {
            var x1 = _testing[i, 0];
            var x2 = _testing[i, 1];
            var y = Forward(model, x1, x2);
            var d = y - _testing[i, 2];
            res += d * d;
        }
        res /= Count;
        return res;
    }
    private static float Sigmoid_f(float x) {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }
}
