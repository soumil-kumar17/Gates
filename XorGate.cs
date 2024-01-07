namespace Gates;
public class XorGate {
    public float Or_w1 { get; set; }
    public float Or_w2 { get; set; }
    public float Or_b { get; set; }
    public float Nand_w1 { get; set; }
    public float Nand_w2 { get; set; }
    public float Nand_b { get; set; }
    public float And_w1 { get; set; }
    public float And_w2 { get; set; }
    public float And_b { get; set; }

    private readonly float[,] _testing = new float[4, 3];
    private const int Count = 4;
    public XorGate(float[,] testing) {
        _testing = testing;
    }
    private XorGate() { }

    public static float Forward(XorGate model, float x1, float x2) {
        float a = Sigmoid_f(model.Or_w1*x1 + model.Or_w2*x2 + model.Or_b);
        float b = Sigmoid_f(model.Nand_w1*x1 + model.Nand_w1 * x2 + model.Nand_b);
        return Sigmoid_f(model.And_w1*a + model.And_w2*b + model.And_b);
    }

    public static void Init_Rand_Xor(XorGate xor) {
        Random random = new();
        xor.Or_w1 = (float)random.NextDouble();
        xor.Or_w2 = (float)random.NextDouble();
        xor.Or_b = (float)random.NextDouble();
        xor.Nand_w1 = (float)random.NextDouble();
        xor.Nand_w2 = (float)random.NextDouble();
        xor.Nand_b = (float)random.NextDouble();
        xor.And_w1 = (float)random.NextDouble();
        xor.And_w2 = (float)random.NextDouble();
        xor.And_b = (float)random.NextDouble();
    }

    public static void Print_Xor(XorGate xor) {
        Console.WriteLine(xor.Or_w1);
        Console.WriteLine(xor.Or_w2);
        Console.WriteLine(xor.Or_b);
        Console.WriteLine(xor.Nand_w1);
        Console.WriteLine(xor.Nand_w2);
        Console.WriteLine(xor.Nand_b);
        Console.WriteLine(xor.And_w1);
        Console.WriteLine(xor.And_w2);
        Console.WriteLine(xor.And_b);
    }

    public XorGate Finite_diff(XorGate m, float eps) {
        XorGate g = new();
        float c = Cost(m);
        float saved;

        saved = m.Or_w1;
        m.Or_w1 += eps;
        g.Or_w1 = (Cost(m) - c) / eps;
        m.Or_w1 = saved;

        saved = m.Or_w2;
        m.Or_w2 += eps;
        g.Or_w2 = (Cost(m) - c) / eps;
        m.Or_w2 = saved;

        saved = m.Or_b;
        m.Or_b += eps;
        g.Or_b = (Cost(m) - c) / eps;
        m.Or_b = saved;

        saved = m.Nand_w1;
        m.Nand_w1 += eps;
        g.Nand_w1 = (Cost(m) - c) / eps;
        m.Nand_w1 = saved;

        saved = m.Nand_w2;
        m.Nand_w2 += eps;
        g.Nand_w2 = (Cost(m) - c) / eps;
        m.Nand_w2 = saved;

        saved = m.Nand_b;
        m.Nand_b += eps;
        g.Nand_b = (Cost(m) - c) / eps;
        m.Nand_b = saved;

        saved = m.And_w1;
        m.And_w1 += eps;
        g.And_w1 = (Cost(m) - c) / eps;
        m.And_w1 = saved;

        saved = m.And_w2;
        m.And_w2 += eps;
        g.And_w2 = (Cost(m) - c) / eps;
        m.And_w2 = saved;

        saved = m.And_b;
        m.And_b += eps;
        g.And_b = (Cost(m) - c) / eps;
        m.And_b = saved;

        return g;
    }

    public static XorGate TrainModel(XorGate m, XorGate g, float rate) {
        m.Or_w1 -= rate * g.Or_w1;
        m.Or_w2 -= rate * g.Or_w2;
        m.Or_b -= rate * g.Or_b;
        m.Nand_w1 -= rate * g.Nand_w1;
        m.Nand_w2 -= rate * g.Nand_w2;
        m.Nand_b -= rate * g.Nand_b;
        m.And_w1 -= rate * g.And_w1;
        m.And_w2 -= rate * g.And_w2;
        m.And_b -= rate * g.And_b;
        return m;
    }

    public float Cost(XorGate model) {
        float res = 0.0f;
        for (int i = 0; i < Count; ++i) {
            float x1 = _testing[i, 0];
            float x2 = _testing[i, 1];
            float y = Forward(model, x1, x2);
            float d = y - _testing[i, 2];
            res += d * d;
        }
        res /= Count;
        return res;
    }
    private static float Sigmoid_f(float x) {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }
}
